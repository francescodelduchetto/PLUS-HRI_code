import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import random
import pandas as pd
from tqdm import tqdm

# Load and Identify "Good" Time Intervals

LABEL_FILE_PATH = '/home/afaris/work/r/FINAL_prepped.csv'
print(f"Loading labels from {LABEL_FILE_PATH}...")

df = pd.read_csv(LABEL_FILE_PATH)
df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True, errors="coerce")
df = df.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)

good_intervals = []
for index, row in df[df['label'] == 'good'].iterrows():
    good_intervals.append((row['start_time'], row['end_time']))

print(f"Found {len(good_intervals)} 'good' time intervals to analyze.")


# Stream Scans and Build DataFrame

SCAN_FILE_PATH = '/home/afaris/work/r/_scan.json' 
print(f"Streaming and indexing scan data from {SCAN_FILE_PATH}...")

all_scans_with_time = []
with open(SCAN_FILE_PATH, 'r') as f:
    for line in tqdm(f, desc="Streaming scans"):
        try:
            scan = json.loads(line)
            ranges = scan.get('ranges', [])
            meta = scan.get('_meta', {})
            inserted_at = meta.get('inserted_at', {})
            date_str = inserted_at.get('$date')
            
            if ranges and date_str and np.count_nonzero(ranges) > 10:
                timestamp = pd.to_datetime(date_str, utc=True, errors='coerce')
                if pd.notna(timestamp):
                    all_scans_with_time.append({'ranges': ranges, 'time': timestamp})
        except json.JSONDecodeError:
            continue

scans_df = pd.DataFrame(all_scans_with_time)
scans_df = scans_df.set_index('time')
scans_df.sort_index(inplace=True)
print(f"Loaded and indexed a total of {len(scans_df)} scans.")


# Build Trajectories from "Good" Intervals
print("Building 'good' trajectories based on CSV time windows...")
good_trajectories = []
for start_time, end_time in tqdm(good_intervals, desc="Matching scans to intervals"):
    scans_in_interval = scans_df.loc[start_time:end_time]
    
    if not scans_in_interval.empty and len(scans_in_interval) >= 20:
        scan_data = scans_in_interval['ranges'].tolist()
        good_trajectories.append(np.array(scan_data, dtype=np.float32))

trajectories_for_plotting = good_trajectories
final_scans_for_clustering = [traj[-1] for traj in trajectories_for_plotting]

print(f"Successfully constructed {len(final_scans_for_clustering)} 'good' trajectories.")

if not final_scans_for_clustering:
    print("\n--- ERROR ---")
    print("Still no matching trajectories were found")
else:
    X_final_scans = StandardScaler().fit_transform(final_scans_for_clustering)

    # Create a "Similarity Map" of Endpoints using t-SNE
    print("Running t-SNE on all final scans...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    endpoint_embedding = tsne.fit_transform(X_final_scans)
    print("t-SNE embedding complete.")

    # Cluster on the t-SNE Map
    N_CLUSTERS = 10
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(endpoint_embedding)
    print(f"Clustering complete. Found {N_CLUSTERS} endpoint clusters.")

    # Create the Individual Plots for Each Cluster
    print("Generating final Individual plots")
    
    STRICTNESS_MULTIPLIER = 1.5

    COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cool', 'spring', 'summer', 'autumn', 'winter']
    plotted_indices = set()

    for i in range(N_CLUSTERS):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) < 5: continue
        original_trajectories_in_cluster = [trajectories_for_plotting[idx] for idx in cluster_indices]
        pca_for_plotting = PCA(n_components=2)
        fig, ax = plt.subplots(figsize=(8, 8))
        all_endpoints_in_plot, plottable_paths, indices_in_plot = [], [], []
        for idx in cluster_indices:
            original_scans = trajectories_for_plotting[idx]
            path_2d = pca_for_plotting.fit_transform(original_scans)
            plottable_paths.append(path_2d)
            all_endpoints_in_plot.append(path_2d[-1])
            indices_in_plot.append(idx)
            
        converging_point = np.mean(all_endpoints_in_plot, axis=0)
        
        
        distances = [np.linalg.norm(ep - converging_point) for ep in all_endpoints_in_plot]
        distance_std = np.std(distances)
        adaptive_threshold = distance_std * STRICTNESS_MULTIPLIER
        
        final_paths_to_plot, final_indices_to_plot = [], []
        for path, original_idx in zip(plottable_paths, indices_in_plot):
            endpoint = path[-1]
            distance_to_center = np.linalg.norm(endpoint - converging_point)
            if distance_to_center <= adaptive_threshold:
                final_paths_to_plot.append(path)
                final_indices_to_plot.append(original_idx)
        
        
        if len(final_paths_to_plot) < 5:
            plt.close(fig)
            continue
        plotted_indices.update(final_indices_to_plot)
        for path_2d in final_paths_to_plot:
            x, y = path_2d[:, 0], path_2d[:, 1]
            cmap = plt.get_cmap(random.choice(COLORMAPS))
            colors = np.linspace(0, 1, len(x))
            for j in range(len(x) - 1):
                ax.plot(x[j:j+2], y[j:j+2], color=cmap(colors[j]), linewidth=0.8, alpha=0.7)
            ax.plot(x[0], y[0], 'o', color='lime', markersize=10, markeredgecolor='black')
            ax.plot(x[-1], y[-1], 'x', color='red', markersize=10, markeredgewidth=2)
        ax.plot(converging_point[0], converging_point[1], 'o', color='black', markersize=25)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_title(f'Cluster {i + 1} ({len(final_paths_to_plot)} trajectories)')
        plt.show()

    # Visualize All Outlier Trajectories
    print(f"\nSuccessfully plotted {len(plotted_indices)} trajectories in clean clusters.")
    all_indices = set(range(len(trajectories_for_plotting)))
    outlier_indices = list(all_indices - plotted_indices)
    print(f"Visualizing the remaining {len(outlier_indices)} outlier trajectories...")
    if outlier_indices:
        fig, ax = plt.subplots(figsize=(8, 8))
        outlier_trajectories = [trajectories_for_plotting[idx] for idx in outlier_indices]
        pca_for_plotting = PCA(n_components=2)
        for original_scans in outlier_trajectories:
            path_2d = pca_for_plotting.fit_transform(original_scans)
            x, y = path_2d[:, 0], path_2d[:, 1]
            cmap = plt.get_cmap(random.choice(COLORMAPS))
            colors = np.linspace(0, 1, len(x))
            for j in range(len(x) - 1):
                ax.plot(x[j:j+2], y[j:j+2], color=cmap(colors[j]), linewidth=0.5, alpha=0.6)
            ax.plot(x[0], y[0], 'o', color='lime', markersize=8, markeredgecolor='black', alpha=0.6)
            ax.plot(x[-1], y[-1], 'x', color='red', markersize=8, markeredgewidth=2, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_title(f'All {len(outlier_indices)} Outlier Trajectories')
        plt.show()