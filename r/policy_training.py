import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gpytorch
from sklearn.decomposition import PCA

# =========================
# Hyperparameters / config
# =========================
MASTER_JSON_PATH = Path("/home/afaris/work/r/master_training_dataset.json")

# Context window for policy input
CONTEXT_LEN = 10
STRIDE = 1
ASOF_TOL = pd.Timedelta("100ms")

# Goal definition approaches
GOAL_APPROACH = "gmm_bic"
MIN_CLUSTER_SIZE = 15
MAX_CLUSTERS = 12
GOAL_RADIUS = 1.6

# Model sizes / training
HIDDEN_DIM = 64
AE_LAYERS = 2
AE_EPOCHS = 40
BATCH_SIZE = 64
LR_AE = 1e-3

# GP settings
GP_INDUCING_MAX = 1000
GP_ITERS = 60
LR_GP = 0.01

# Simulation
MAX_SIM_STEPS = 50
MIN_TRAJ_LENGTH = 10

# Dynamics
FIT_LATENT_DYNAMICS = True
RIDGE_ALPHA = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def compute_velocities(pose_df: pd.DataFrame) -> pd.DataFrame:
    pose_df = pose_df.copy()
    dt = pose_df.index.to_series().diff().dt.total_seconds()
    dxy = pose_df[["x", "y"]].diff()
    pose_df["linear_velocity"] = (np.linalg.norm(dxy, axis=1) / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    quats = pose_df[["qx", "qy", "qz", "qw"]].values
    yaws = R.from_quat(quats).as_euler("xyz", degrees=False)[:, 2]
    dyaw = pd.Series(yaws, index=pose_df.index).diff()
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    pose_df["angular_velocity_z"] = (dyaw / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pose_df

def ema_smooth(series: pd.Series, span: int = 5) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def build_traj_frames(traj_obj: dict):
    # Poses
    poses = []
    for p in traj_obj.get("poses", []):
        ts = pd.to_datetime(p.get("timestamp"), utc=True, errors="coerce")
        if ts is pd.NaT:
            continue
        pos = p.get("position", {})
        ori = p.get("orientation", {})
        poses.append({
            "time": ts,
            "x": float(pos.get("x", np.nan)),
            "y": float(pos.get("y", np.nan)),
            "qx": float(ori.get("x", 0.0)),
            "qy": float(ori.get("y", 0.0)),
            "qz": float(ori.get("z", 0.0)),
            "qw": float(ori.get("w", 1.0)),
        })
    if not poses:
        return None, None
    pose_df = pd.DataFrame(poses).dropna(subset=["x", "y"]).set_index("time").sort_index()
    if pose_df.shape[0] < MIN_TRAJ_LENGTH:
        return None, None

    pose_df = compute_velocities(pose_df)
    pose_df["linear_velocity"] = ema_smooth(pose_df["linear_velocity"], span=5).fillna(0.0)
    pose_df["angular_velocity_z"] = ema_smooth(pose_df["angular_velocity_z"], span=5).fillna(0.0)

    # Scans
    scans = []
    for s in traj_obj.get("scans", []):
        ts = pd.to_datetime(s.get("timestamp"), utc=True, errors="coerce")
        ranges = s.get("ranges")
        if ts is pd.NaT or not isinstance(ranges, list) or len(ranges) == 0:
            continue
        scans.append({"time": ts, "ranges": np.asarray(ranges, dtype=np.float32)})
    if not scans:
        return None, None

    scans_df = pd.DataFrame(scans).set_index("time").sort_index()
    dims = scans_df["ranges"].map(lambda a: a.shape[0]).value_counts()
    if dims.empty:
        return None, None
    common_dim = dims.idxmax()
    scans_df = scans_df[scans_df["ranges"].map(lambda a: a.shape[0] == common_dim)]
    if scans_df.empty:
        return None, None

    vel_df = pd.merge_asof(
        scans_df.reset_index().sort_values("time"),
        pose_df[["linear_velocity", "angular_velocity_z"]].reset_index().sort_values("time"),
        on="time", direction="nearest", tolerance=ASOF_TOL
    ).dropna().set_index("time").sort_index()

    if vel_df.empty:
        return None, None

    return scans_df.loc[vel_df.index], vel_df[["linear_velocity", "angular_velocity_z"]]

def make_policy_samples(scans_df: pd.DataFrame, vel_df: pd.DataFrame, context_len: int, stride: int):
    times = vel_df.index
    X_scans, X_prev_vel, Y_next_vel = [], [], []
    
    for start_idx in range(0, len(times) - context_len, stride):
        context_times = times[start_idx:start_idx + context_len]
        next_time = times[start_idx + context_len]
        
        all_times = context_times.append(pd.Index([next_time]))
        if not (np.all(np.diff(all_times.view(np.int64)) > 0)):
            continue
            
        context_scans = np.vstack(scans_df.loc[context_times, "ranges"].values)
        prev_vel = vel_df.loc[context_times[-1], ["linear_velocity", "angular_velocity_z"]].values
        next_vel = vel_df.loc[next_time, ["linear_velocity", "angular_velocity_z"]].values
        
        X_scans.append(context_scans.astype(np.float32))
        X_prev_vel.append(prev_vel.astype(np.float32))
        Y_next_vel.append(next_vel.astype(np.float32))
    
    return X_scans, X_prev_vel, Y_next_vel

# =========================
# Model definitions
# =========================
class PolicyEncoder(nn.Module):
    def __init__(self, scan_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.scan_encoder = nn.LSTM(scan_dim, hidden_dim, n_layers, batch_first=True)
        self.vel_fc = nn.Linear(2, hidden_dim // 4)
        self.fusion_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
    def forward(self, scans, prev_vel):
        # scans: [B, T, D], prev_vel: [B, 2]
        _, (scan_h, _) = self.scan_encoder(scans)
        scan_features = scan_h[-1]  # [B, H]
        
        vel_features = torch.relu(self.vel_fc(prev_vel))  # [B, H//4]
        
        combined = torch.cat([scan_features, vel_features], dim=1)  # [B, H + H//4]
        return torch.relu(self.fusion_fc(combined))  # [B, H]

class PolicyDataset(Dataset):
    def __init__(self, X_scans, X_prev_vel, Y_next_vel):
        self.scans = [torch.tensor(x, dtype=torch.float32) for x in X_scans]
        self.prev_vel = torch.tensor(np.array(X_prev_vel), dtype=torch.float32)
        self.next_vel = torch.tensor(np.array(Y_next_vel), dtype=torch.float32)
        assert len(self.scans) == len(self.prev_vel) == len(self.next_vel)

    def __len__(self): 
        return len(self.scans)
        
    def __getitem__(self, idx): 
        return self.scans[idx], self.prev_vel[idx], self.next_vel[idx]

class MultiOutSparseGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks=2, num_latents=2):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_latents])
        )
        base_vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        vs = gpytorch.variational.LMCVariationalStrategy(
            base_vs, num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1
        )
        super().__init__(vs)
        self.mean_module  = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def extract_trajectory_endpoints(traj_frames, policy_encoder, scan_scaler, prev_vel_scaler):
    endpoints = []
    
    for traj_id, scans_df, vel_df in traj_frames:
        if len(vel_df) < CONTEXT_LEN:
            continue
            
        end_times = vel_df.index[-CONTEXT_LEN:]
        end_scans = np.vstack(scans_df.loc[end_times, "ranges"].values)
        end_scans = scan_scaler.transform(end_scans)
        
        prev_vel = prev_vel_scaler.transform(
            vel_df.iloc[-2][["linear_velocity", "angular_velocity_z"]].values.reshape(1, -1)
        )[0]
        
        with torch.no_grad():
            scans_t = torch.tensor(end_scans, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_vel_t = torch.tensor(prev_vel, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            endpoint_z = policy_encoder(scans_t, prev_vel_t)[0].cpu().numpy()
            
        endpoints.append({
            'traj_id': traj_id,
            'latent': endpoint_z,
            'trajectory_length': len(vel_df)
        })
    
    return endpoints

def find_optimal_unstuck_regions(endpoints, approach="gmm_bic"):
    latent_points = np.vstack([ep['latent'] for ep in endpoints])
    
    if approach == "mean":
        goal_center = np.mean(latent_points, axis=0)
        goal_std = np.std(latent_points, axis=0)
        
        return {
            'type': 'mean',
            'centers': [goal_center],
            'labels': np.zeros(len(endpoints), dtype=int),
            'n_clusters': 1
        }
    
    elif approach == "gmm_bic":
        n_components_range = range(1, min(MAX_CLUSTERS + 1, len(endpoints) // MIN_CLUSTER_SIZE + 1))
        if not n_components_range:
             n_components_range = [1]
        bic_scores = []
        aic_scores = []
        models = []
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
            gmm.fit(latent_points)
            bic_scores.append(gmm.bic(latent_points))
            aic_scores.append(gmm.aic(latent_points))
            models.append(gmm)
        
        optimal_idx = np.argmin(bic_scores)
        optimal_n = n_components_range[optimal_idx]
        best_gmm = models[optimal_idx]
        
        print(f"Optimal clusters: {optimal_n} (BIC: {bic_scores[optimal_idx]:.1f})")
        
        labels = best_gmm.predict(latent_points)
        cluster_sizes = np.bincount(labels)
        
        valid_clusters = np.where(cluster_sizes >= MIN_CLUSTER_SIZE)[0]
        if len(valid_clusters) == 0:
            print("Warning: No clusters meet minimum size, using single mean goal")
            return find_optimal_unstuck_regions(endpoints, "mean")
        
        valid_labels = np.full_like(labels, -1)
        valid_centers = []
        for new_idx, old_idx in enumerate(valid_clusters):
            mask = labels == old_idx
            valid_labels[mask] = new_idx
            valid_centers.append(best_gmm.means_[old_idx])
        
        valid_mask = valid_labels >= 0
        valid_labels = valid_labels[valid_mask]
        
        print(f"Valid clusters: {len(valid_centers)} (removed {optimal_n - len(valid_centers)} small clusters)")
        for i, cluster_id in enumerate(valid_clusters):
            size = cluster_sizes[cluster_id]
            print(f"Cluster {i}: {size} trajectories")
        
        return {
            'type': 'gmm_bic',
            'centers': valid_centers,
            'labels': valid_labels,
            'valid_mask': valid_mask,
            'n_clusters': len(valid_centers),
            'gmm': best_gmm
        }
    
    else:
        raise ValueError(f"Unknown approach: {approach}")

def create_cluster_specific_datasets(traj_frames, goal_info, policy_encoder, scan_scaler, prev_vel_scaler, y_scaler):
    
    if goal_info['type'] == 'mean':
        return create_single_cluster_dataset(traj_frames, goal_info['centers'][0], 
                                           policy_encoder, scan_scaler, prev_vel_scaler, y_scaler)
    
    elif goal_info['type'] == 'gmm_bic':
        cluster_datasets = {}
        
        endpoints = extract_trajectory_endpoints(traj_frames, policy_encoder, scan_scaler, prev_vel_scaler)
        valid_mask = goal_info['valid_mask']
        valid_labels = goal_info['labels']
        
        traj_clusters = {}
        valid_traj_indices = np.where(valid_mask)[0]
        for i, label in enumerate(valid_labels):
            traj_idx = valid_traj_indices[i]
            if label not in traj_clusters:
                traj_clusters[label] = []
            traj_clusters[label].append(traj_frames[traj_idx])

        for cluster_id in range(goal_info['n_clusters']):
            if cluster_id in traj_clusters:
                cluster_trajs = traj_clusters[cluster_id]
                goal_center = goal_info['centers'][cluster_id]
                
                print(f"Cluster {cluster_id}: {len(cluster_trajs)} trajectories")
                dataset = create_single_cluster_dataset(cluster_trajs, goal_center,
                                                      policy_encoder, scan_scaler, prev_vel_scaler, y_scaler)
                cluster_datasets[cluster_id] = dataset
        
        return cluster_datasets

def create_single_cluster_dataset(traj_frames, goal_center, policy_encoder, scan_scaler, prev_vel_scaler, y_scaler):
    X_scans, X_prev_vel, Y_next_vel = [], [], []
    
    MAX_SAMPLES_PER_TRAJ = 30
    
    for _, scans_df, vel_df in traj_frames:
        X_s, X_p, Y_n = make_policy_samples(scans_df, vel_df, CONTEXT_LEN, STRIDE)
        
        n_samples = min(len(X_s), MAX_SAMPLES_PER_TRAJ)
        if n_samples > 0:
            if len(X_s) > MAX_SAMPLES_PER_TRAJ:
                sample_idx = np.random.choice(len(X_s), n_samples, replace=False)
                X_s = [X_s[i] for i in sample_idx]
                X_p = [X_p[i] for i in sample_idx]
                Y_n = [Y_n[i] for i in sample_idx]
            
            X_scans.extend(X_s)
            X_prev_vel.extend(X_p)
            Y_next_vel.extend(Y_n)
    
    X_scans_norm = [scan_scaler.transform(seq) for seq in X_scans]
    X_prev_vel_std = prev_vel_scaler.transform(np.array(X_prev_vel))
    Y_next_vel_std = y_scaler.transform(np.array(Y_next_vel))
    
    features = []
    policy_encoder.eval()
    with torch.no_grad():
        for i, scans in enumerate(X_scans_norm):
            scans_t = torch.tensor(scans, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_vel_t = torch.tensor(X_prev_vel_std[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            feat = policy_encoder(scans_t, prev_vel_t)[0].cpu().numpy()
            features.append(feat)
    
    Z = np.vstack(features)
    
    return {
        'features': Z,
        'targets': Y_next_vel_std,
        'goal_center': goal_center,
        'n_samples': len(Z)
    }

def train_cluster_gp(dataset, cluster_id):
    print(f"Training GP")
    
    X_tensor = torch.tensor(dataset['features'], dtype=torch.float32, device=DEVICE)
    Y_tensor = torch.tensor(dataset['targets'], dtype=torch.float32, device=DEVICE)
    
    num_inducing = min(GP_INDUCING_MAX, len(X_tensor))
    perm = torch.randperm(len(X_tensor))[:num_inducing]
    inducing = X_tensor[perm]
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(DEVICE)
    gp_model = MultiOutSparseGP(inducing).to(DEVICE)
    
    gp_model.train()
    likelihood.train()
    gp_opt = torch.optim.Adam(gp_model.parameters(), lr=LR_GP)
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=len(Y_tensor))
    
    batch_size = 512
    n_batches = (len(X_tensor) + batch_size - 1) // batch_size
    
    for epoch in range(GP_ITERS):
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_tensor))
            
            X_batch = X_tensor[start_idx:end_idx]
            Y_batch = Y_tensor[start_idx:end_idx]
            
            gp_opt.zero_grad()
            out = gp_model(X_batch)
            loss = -mll(out, Y_batch)
            loss.backward()
            gp_opt.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"  Cluster {cluster_id} Epoch {epoch+1}/{GP_ITERS} | Loss: {avg_loss:.4f}")
    
    return gp_model, likelihood

def simulate_with_detailed_logging(start_context_scans, start_prev_vel, 
                                   goal_info, cluster_gps, cluster_likelihoods,
                                   policy_encoder, scan_scaler, prev_vel_scaler, y_scaler,
                                   pca_3d, adaptive_goal_radius, latent_dynamics_model,
                                   max_steps=MAX_SIM_STEPS, verbose=True):
    
    policy_encoder.eval()
    
    with torch.no_grad():
        scans_t = torch.tensor(start_context_scans, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_vel_t = torch.tensor(start_prev_vel, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        current_z = policy_encoder(scans_t, prev_vel_t)[0].cpu().numpy()

    if goal_info['type'] == 'mean':
        target_cluster = 0
        target_goal = goal_info['centers'][0]
    else:
        distances = [np.linalg.norm(current_z - center) for center in goal_info['centers']]
        target_cluster = np.argmin(distances)
        target_goal = goal_info['centers'][target_cluster]
    
    initial_dist = np.linalg.norm(current_z - target_goal)
    if verbose:
        print(f"Assigned to cluster {target_cluster}, initial distance to goal: {initial_dist:.3f}")
        print(f"Goal radius threshold: {adaptive_goal_radius:.3f}")
    
    latent_trajectory = [current_z]
    latent_3d_trajectory = [pca_3d.transform([current_z])[0]]
    velocity_commands = []
    distances_to_goal = [initial_dist]
    
    if goal_info['type'] == 'mean':
        gp_model, likelihood = cluster_gps[0], cluster_likelihoods[0]
    else:
        gp_model, likelihood = cluster_gps[target_cluster], cluster_likelihoods[target_cluster]

    gp_model.eval()
    likelihood.eval()
    
    best_dist = initial_dist
    stuck_counter = 0
    no_progress_counter = 0
    
    if verbose:
        print("\n=== Starting Simulation Steps ===")
        print("Step | Lin Vel | Ang Vel | Dist to Goal | Progress")
        print("-" * 55)
    
    with torch.no_grad():
        for step in range(max_steps):
            if verbose and (initial_dist < adaptive_goal_radius and step == 0):
                print("Note: Starting position is within goal region, but continuing simulation...")
            
            dist_to_goal = np.linalg.norm(current_z - target_goal)
            
            if dist_to_goal < adaptive_goal_radius and step > 0:
                if verbose:
                    print(f"\n✓ REACHED GOAL in {step} steps! Final distance: {dist_to_goal:.3f}")
                return {
                    'latent_trajectory': latent_trajectory,
                    'latent_3d_trajectory': latent_3d_trajectory,
                    'velocity_commands': velocity_commands,
                    'distances_to_goal': distances_to_goal,
                    'success': True,
                    'steps': step
                }
            
            z_t = torch.tensor(current_z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            gp_action_std = likelihood(gp_model(z_t)).mean[0].cpu().numpy()
            gp_action = y_scaler.inverse_transform(gp_action_std.reshape(1, -1))[0]
            
            action = gp_action
            action_type = "gp"

            action = np.array([
                np.clip(action[0], -0.5, 0.5),
                np.clip(action[1], -1.0, 1.0)
            ])
            
            velocity_commands.append({
                'step': step + 1,
                'linear_vel': action[0],
                'angular_vel': action[1],
                'distance': dist_to_goal,
                'action_type': action_type
            })
            
            if verbose and (step % 5 == 0 or stuck_counter > 5):
                progress = (initial_dist - dist_to_goal) / initial_dist * 100
                print(f"{step+1:4d} | {action[0]:7.4f} | {action[1]:8.4f} | {dist_to_goal:12.3f} | {progress:6.1f}% ({action_type})")
            
            dynamics_input = np.hstack([current_z, y_scaler.transform(action.reshape(1, -1))[0]])
            current_z = latent_dynamics_model.predict(dynamics_input.reshape(1, -1))[0]
            
            latent_trajectory.append(current_z)
            latent_3d_trajectory.append(pca_3d.transform([current_z])[0])
            distances_to_goal.append(np.linalg.norm(current_z - target_goal))
            
            if dist_to_goal < best_dist - 0.01:
                best_dist = dist_to_goal
                stuck_counter = 0
                no_progress_counter = 0
            else:
                stuck_counter += 1
                if abs(dist_to_goal - best_dist) < 0.005:
                    no_progress_counter += 1
            
            if no_progress_counter > 20:
                if verbose:
                    print(f"\n✗ Terminating: No progress for 20 steps. Final distance: {dist_to_goal:.3f}")
                break
    
    if verbose:
        print(f"\n✗ Max steps reached. Final distance: {distances_to_goal[-1]:.3f}")
    return {
        'latent_trajectory': latent_trajectory,
        'latent_3d_trajectory': latent_3d_trajectory,
        'velocity_commands': velocity_commands,
        'distances_to_goal': distances_to_goal,
        'success': False,
        'steps': len(velocity_commands)
    }

# =========================
# Main training pipeline
# =========================

print("=== Load master JSON ===")
with open(MASTER_JSON_PATH, "r") as f:
    master = json.load(f)
traj_keys = sorted(master.keys(), key=lambda k: int(k))
print(f"Found {len(traj_keys)} trajectories.")

traj_frames = []
lidar_dim = None
skipped = 0
for k in tqdm(traj_keys):
    scans_df, vel_df = build_traj_frames(master[k])
    if scans_df is None or vel_df is None:
        skipped += 1
        continue
    d = scans_df.iloc[0]["ranges"].shape[0]
    if lidar_dim is None:
        lidar_dim = d
    elif lidar_dim != d:
        continue
    traj_frames.append((k, scans_df, vel_df))
print(f"Usable trajectories: {len(traj_frames)} (skipped {skipped}). LIDAR dim = {lidar_dim}")

print("=== Train/test split by trajectory ===")
traj_ids = [k for (k, _, _) in traj_frames]
train_ids, test_ids = train_test_split(traj_ids, test_size=0.2, random_state=RANDOM_STATE)
train_trajs = [(k, s, v) for (k, s, v) in traj_frames if k in train_ids]
test_trajs  = [(k, s, v) for (k, s, v) in traj_frames if k in test_ids]
print(f"Train trajs: {len(train_trajs)}, Test trajs: {len(test_trajs)}")

X_scans_train, X_prev_vel_train, Y_next_vel_train = [], [], []
for _, scans_df, vel_df in train_trajs:
    X_s, X_p, Y_n = make_policy_samples(scans_df, vel_df, CONTEXT_LEN, STRIDE)
    X_scans_train.extend(X_s)
    X_prev_vel_train.extend(X_p)
    Y_next_vel_train.extend(Y_n)
print(f"Train samples: {len(X_scans_train)}")

train_scans_stack = np.vstack(X_scans_train)
scan_scaler = StandardScaler().fit(train_scans_stack)
X_scans_train_norm = [scan_scaler.transform(context) for context in X_scans_train]
Y_train = np.array(Y_next_vel_train, dtype=np.float32)
y_scaler = StandardScaler().fit(Y_train)
Y_train_std = y_scaler.transform(Y_train)
prev_vel_scaler = StandardScaler().fit(np.array(X_prev_vel_train))
X_prev_vel_train_std = prev_vel_scaler.transform(np.array(X_prev_vel_train))

print("=== Policy Encoder ===")
policy_encoder = PolicyEncoder(lidar_dim, HIDDEN_DIM, AE_LAYERS).to(DEVICE)
velocity_head = nn.Linear(HIDDEN_DIM, 2).to(DEVICE)
policy_opt = torch.optim.Adam(
    list(policy_encoder.parameters()) + list(velocity_head.parameters()), 
    lr=LR_AE
)
policy_loss = nn.MSELoss()
policy_ds = PolicyDataset(X_scans_train_norm, X_prev_vel_train_std, Y_train_std)
policy_dl = DataLoader(policy_ds, batch_size=BATCH_SIZE, shuffle=True)
policy_encoder.train()
velocity_head.train()
for epoch in range(AE_EPOCHS):
    total_loss = 0.0
    for scans, prev_vel, next_vel in policy_dl:
        scans, prev_vel, next_vel = scans.to(DEVICE), prev_vel.to(DEVICE), next_vel.to(DEVICE)
        policy_opt.zero_grad()
        features = policy_encoder(scans, prev_vel)
        pred = velocity_head(features)
        loss = policy_loss(pred, next_vel)
        loss.backward()
        policy_opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Basic Policy Epoch {epoch+1}/{AE_EPOCHS} | Loss: {total_loss / len(policy_dl):.6f}")
print("Basic policy encoder trained.")

endpoints = extract_trajectory_endpoints(train_trajs, policy_encoder, scan_scaler, prev_vel_scaler)
print(f"Extracted {len(endpoints)} trajectory endpoints")

goal_info = find_optimal_unstuck_regions(endpoints, approach=GOAL_APPROACH)

cluster_datasets = create_cluster_specific_datasets(
    train_trajs, goal_info, policy_encoder, scan_scaler, prev_vel_scaler, y_scaler
)

print("=== Train GP ===")
cluster_gps = {}
cluster_likelihoods = {}
if goal_info['type'] == 'mean' and isinstance(cluster_datasets, dict) and 'features' in cluster_datasets:
    dataset = cluster_datasets
    gp_model, likelihood = train_cluster_gp(dataset, 0)
    cluster_gps[0] = gp_model
    cluster_likelihoods[0] = likelihood
else:
    for cluster_id, dataset in cluster_datasets.items():
        gp_model, likelihood = train_cluster_gp(dataset, cluster_id)
        cluster_gps[cluster_id] = gp_model
        cluster_likelihoods[cluster_id] = likelihood

print("\n=== Training Latent Dynamics Model ===")
latent_dynamics_model = None
if FIT_LATENT_DYNAMICS:
    X_dyn, Y_dyn = [], []
    policy_encoder.eval()
    with torch.no_grad():
        for i in range(len(X_scans_train_norm) - 1):
            scans_t = torch.tensor(X_scans_train_norm[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_vel_t = torch.tensor(X_prev_vel_train_std[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            z_t = policy_encoder(scans_t, prev_vel_t)[0].cpu().numpy()
            a_t = Y_train_std[i]
            scans_t1 = torch.tensor(X_scans_train_norm[i+1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_vel_t1 = torch.tensor(X_prev_vel_train_std[i+1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            z_t1 = policy_encoder(scans_t1, prev_vel_t1)[0].cpu().numpy()
            X_dyn.append(np.hstack([z_t, a_t]))
            Y_dyn.append(z_t1)
    X_dyn = np.array(X_dyn)
    Y_dyn = np.array(Y_dyn)
    latent_dynamics_model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    latent_dynamics_model.fit(X_dyn, Y_dyn)
    score = latent_dynamics_model.score(X_dyn, Y_dyn)
    print(f"Latent dynamics model trained. R² score: {score:.4f}")


print(f"\n{'='*25} DETAILED MODEL PERFORMANCE EVALUATION (ON TEST SET) {'='*25}")


print("Creating test set samples for evaluation...")
X_scans_test, X_prev_vel_test, Y_next_vel_test = [], [], []
for _, scans_df, vel_df in test_trajs:
    X_s, X_p, Y_n = make_policy_samples(scans_df, vel_df, CONTEXT_LEN, STRIDE)
    X_scans_test.extend(X_s)
    X_prev_vel_test.extend(X_p)
    Y_next_vel_test.extend(Y_n)


X_scans_test_norm = [scan_scaler.transform(context) for context in X_scans_test]
X_prev_vel_test_std = prev_vel_scaler.transform(np.array(X_prev_vel_test))
Y_test = np.array(Y_next_vel_test)
Y_test_std = y_scaler.transform(Y_test)

print("\n--- 1. Policy Encoder Performance ---")
policy_encoder.eval()
velocity_head.eval()
encoder_preds_std = []
with torch.no_grad():
    for i in range(len(X_scans_test_norm)):
        scans_t = torch.tensor(X_scans_test_norm[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_vel_t = torch.tensor(X_prev_vel_test_std[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        features = policy_encoder(scans_t, prev_vel_t)
        pred = velocity_head(features)[0].cpu().numpy()
        encoder_preds_std.append(pred)
encoder_preds_std = np.array(encoder_preds_std)
encoder_preds = y_scaler.inverse_transform(encoder_preds_std)

encoder_r2 = r2_score(Y_test, encoder_preds)
encoder_mse = mean_squared_error(Y_test, encoder_preds)
print(f"R² Score: {encoder_r2:.4f}")
print(f"Mean Squared Error (m/s, rad/s): {encoder_mse:.4f}")

print("\n--- 2. Latent Dynamics Model Performance ---")
X_dyn_test, Y_dyn_test = [], []
with torch.no_grad():
    for i in range(len(X_scans_test_norm) - 1):
        scans_t = torch.tensor(X_scans_test_norm[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_vel_t = torch.tensor(X_prev_vel_test_std[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z_t = policy_encoder(scans_t, prev_vel_t)[0].cpu().numpy()
        a_t = Y_test_std[i]
        scans_t1 = torch.tensor(X_scans_test_norm[i+1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_vel_t1 = torch.tensor(X_prev_vel_test_std[i+1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z_t1 = policy_encoder(scans_t1, prev_vel_t1)[0].cpu().numpy()
        X_dyn_test.append(np.hstack([z_t, a_t]))
        Y_dyn_test.append(z_t1)
X_dyn_test = np.array(X_dyn_test)
Y_dyn_test = np.array(Y_dyn_test)

dynamics_r2 = latent_dynamics_model.score(X_dyn_test, Y_dyn_test)
dynamics_mse = mean_squared_error(Y_dyn_test, latent_dynamics_model.predict(X_dyn_test))
print(f"R² Score: {dynamics_r2:.4f}")
print(f"Mean Squared Error (latent units): {dynamics_mse:.4f}")

# Evaluate GP Policy
print("\n--- 3. GP Policy Performance ---")
test_features = []
with torch.no_grad():
    for i in range(len(X_scans_test_norm)):
        scans_t = torch.tensor(X_scans_test_norm[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_vel_t = torch.tensor(X_prev_vel_test_std[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        feat = policy_encoder(scans_t, prev_vel_t)[0].cpu().numpy()
        test_features.append(feat)
test_features = np.array(test_features)

gp_preds_std = []
for cluster_id, gp_model in cluster_gps.items():
    likelihood = cluster_likelihoods[cluster_id]
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred_dist = likelihood(gp_model(torch.tensor(test_features, dtype=torch.float32).to(DEVICE)))
        gp_preds_std.append(pred_dist.mean.cpu().numpy())

# Assuming one cluster for simplicity as per GMM results, otherwise needs assignment logic
gp_preds_std = gp_preds_std[0]
gp_preds = y_scaler.inverse_transform(gp_preds_std)
gp_r2 = r2_score(Y_test, gp_preds)
gp_mse = mean_squared_error(Y_test, gp_preds)
print(f"R² Score: {gp_r2:.4f}")
print(f"Mean Squared Error (m/s, rad/s): {gp_mse:.4f}")


test_endpoints = extract_trajectory_endpoints(test_trajs, policy_encoder, scan_scaler, prev_vel_scaler)
test_goal_info = find_optimal_unstuck_regions(test_endpoints, approach=GOAL_APPROACH)

if test_endpoints:
    if test_goal_info['type'] == 'mean':
        test_goal_center = test_goal_info['centers'][0]
        test_distances_to_center = [np.linalg.norm(ep['latent'] - test_goal_center) for ep in test_endpoints]
    else:
        test_labels = test_goal_info.get('labels', np.zeros(len(test_endpoints), dtype=int))
        test_valid_mask = test_goal_info.get('valid_mask', np.ones(len(test_endpoints), dtype=bool))
        test_centers = test_goal_info['centers']
        test_distances_to_center = []
        valid_test_endpoints = np.array(test_endpoints)[test_valid_mask]
        for i, ep in enumerate(valid_test_endpoints):
            cluster_id = test_labels[i]
            if cluster_id >= 0 and cluster_id < len(test_centers):
                 dist = np.linalg.norm(ep['latent'] - test_centers[cluster_id])
                 test_distances_to_center.append(dist)
    
    test_mean_dist = np.mean(test_distances_to_center)
    TEST_ADAPTIVE_GOAL_RADIUS = test_mean_dist * 1.10
else:
    TEST_ADAPTIVE_GOAL_RADIUS = 1.0
    print("No test endpoints found, using default goal radius.")

print(f"\n{'='*25} EVALUATING POLICY ON FULL TEST SET {'='*25}")
print(f"Running simulation on all {len(test_trajs)} test trajectories...")

all_results_info = []
pca_3d = PCA(n_components=3)
all_endpoints_latent = np.vstack([ep['latent'] for ep in endpoints] + [ep['latent'] for ep in test_endpoints if test_endpoints])
endpoints_3d = pca_3d.fit_transform(all_endpoints_latent)

for i, test_traj in enumerate(tqdm(test_trajs, desc="Test Set Evaluation")):
    traj_id, scans_df, vel_df = test_traj
    if len(vel_df) < CONTEXT_LEN + 1: continue
    start_times = vel_df.index[:CONTEXT_LEN]
    start_scans = np.vstack(scans_df.loc[start_times, "ranges"].values)
    start_scans_norm = scan_scaler.transform(start_scans)
    start_vel = prev_vel_scaler.transform(
        vel_df.iloc[CONTEXT_LEN-1][["linear_velocity", "angular_velocity_z"]].values.reshape(1, -1)
    )[0]
    result = simulate_with_detailed_logging(
        start_scans_norm, start_vel,
        test_goal_info, cluster_gps, cluster_likelihoods,
        policy_encoder, scan_scaler, prev_vel_scaler, y_scaler,
        pca_3d, TEST_ADAPTIVE_GOAL_RADIUS, latent_dynamics_model,
        max_steps=50, verbose=False
    )
    all_results_info.append({'result': result, 'index': i, 'traj_id': traj_id})

all_results = [r['result'] for r in all_results_info]
successful_runs = [r for r in all_results if r['success']]
failed_runs = [r for r in all_results if not r['success']]
total_runs = len(all_results)
success_rate = (len(successful_runs) / total_runs) * 100 if total_runs > 0 else 0
avg_steps_to_success = np.mean([r['steps'] for r in successful_runs]) if successful_runs else 0
improvements = []
for r in all_results:
    if r['distances_to_goal'] and r['distances_to_goal'][0] > 0.001:
        improvements.append((r['distances_to_goal'][0] - r['distances_to_goal'][-1]) / r['distances_to_goal'][0])
avg_improvement = np.mean(improvements) * 100 if improvements else 0

print(f"\n\n{'='*60}")
print(f"{'POLICY PERFORMANCE ON TEST SET':^60}")
print(f"{'='*60}")
print(f"Total Test Trajectories: {total_runs}")
print("-" * 60)
print(f"✅ Success Rate: {success_rate:.2f}% ({len(successful_runs)}/{total_runs})")
print(f"❌ Failure Rate: {100 - success_rate:.2f}% ({len(failed_runs)}/{total_runs})")
print("-" * 60)
print(f"⏱️ Average Steps to Success: {avg_steps_to_success:.2f} steps")
print(f"{'='*60}")
print(f"\n{'='*25} VISUALIZING TOP 5 LONGEST SUCCESSFUL RUNS {'='*25}")

successful_runs_info = [r for r in all_results_info if r['result']['success']]
sorted_successful_runs = sorted(successful_runs_info, key=lambda x: x['result']['steps'], reverse=True)
top_n_to_plot = 5
runs_to_plot = sorted_successful_runs[:top_n_to_plot]

if not runs_to_plot:
    print("No successful runs found to visualize.")
else:
    for i, run_info in enumerate(runs_to_plot):
        result = run_info['result']
        traj_id = run_info['traj_id']
        
        print(f"\n--- Plotting Rank {i+1}: Trajectory {traj_id} ({result['steps']} steps) ---")

        fig = plt.figure(figsize=(16, 10))
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_3d.scatter(endpoints_3d[:, 0], endpoints_3d[:, 1], endpoints_3d[:, 2], c='lightgray', alpha=0.1, s=5, label='All Endpoints')
        goal_centers_3d = pca_3d.transform(np.array(test_goal_info['centers']))
        for j, goal_3d in enumerate(goal_centers_3d):
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_sphere = goal_3d[0] + TEST_ADAPTIVE_GOAL_RADIUS * np.cos(u)*np.sin(v)
            y_sphere = goal_3d[1] + TEST_ADAPTIVE_GOAL_RADIUS * np.sin(u)*np.sin(v)
            z_sphere = goal_3d[2] + TEST_ADAPTIVE_GOAL_RADIUS * np.cos(v)
            ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color='yellow', label=f'Goal Region {j}' if j == 0 else "")
            ax_3d.scatter([goal_3d[0]], [goal_3d[1]], [goal_3d[2]], c='black', s=200, marker='X', edgecolor='gold', linewidth=2, label='Goal Center(s)' if j == 0 else "")
        if len(result['latent_3d_trajectory']) > 0:
            traj_3d = np.array(result['latent_3d_trajectory'])
            ax_3d.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], 'b-', alpha=0.6, linewidth=2, label='Simulated Path')
            ax_3d.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2], c='cyan', s=150, marker='o', edgecolor='blue', linewidth=2, label='Start Point')
            ax_3d.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2], c='lime', s=150, marker='*', edgecolor='darkred', linewidth=2, label='End Point (Success)')
        ax_3d.set_title('3D Visualization of Latent Space')
        ax_3d.legend(loc='upper right')
        ax_dist = fig.add_subplot(222)
        steps = range(len(result['distances_to_goal']))
        ax_dist.plot(steps, result['distances_to_goal'], 'b-', linewidth=2)
        ax_dist.axhline(y=TEST_ADAPTIVE_GOAL_RADIUS, color='green', linestyle='--', label=f'Goal Radius ({TEST_ADAPTIVE_GOAL_RADIUS:.3f})')
        ax_dist.fill_between(steps, 0, TEST_ADAPTIVE_GOAL_RADIUS, alpha=0.2, color='green')
        ax_dist.set_title('Convergence Progress')
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend()
        ax_vel = fig.add_subplot(224)
        if len(result['velocity_commands']) > 0:
            vel_data = pd.DataFrame(result['velocity_commands'])
            ax_vel.plot(vel_data['step'], vel_data['linear_vel'], 'b-', label='Linear', linewidth=2)
            ax_vel.plot(vel_data['step'], vel_data['angular_vel'], 'r-', label='Angular', linewidth=2)
            ax_vel.set_title('Velocity Commands Over Time')
            ax_vel.grid(True, alpha=0.3)
            ax_vel.legend()
        plt.suptitle(f'Unstuck Policy Simulation - Trajectory {traj_id}\nResult: SUCCESS in {result["steps"]} steps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_filename = f'longest_successful_trajectory_rank{i+1}_traj_{traj_id}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to '{plot_filename}'")
        plt.show()

        print(f"\n--- Velocity Command Log for Trajectory {traj_id} ---")
        if len(result['velocity_commands']) > 0:
            vel_df_log = pd.DataFrame(result['velocity_commands'])
            print(vel_df_log.to_string())
        else:
            print("No velocity commands for this run.")