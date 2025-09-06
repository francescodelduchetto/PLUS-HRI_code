FROM palroboticssl/tiago_tutorials:noetic

# SHELL ["/bin/bash", "-c"] 

# Config
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV ROS_DISTRO noetic

# Minimal setup
RUN apt-get update && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales


# Setup ROS sources and keys
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN apt-key del "C1CF 6E31 E6BA DE88 68B1 72B4 F42E D6FB AB17 C654"
RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654


# # Install ROS
# RUN rm /etc/apt/sources.list.d/* && \
#     DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends && \
#     DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common lsb-release curl wget apt-transport-https git tmux terminator nano htop net-tools iputils-ping sshpass iproute2 --no-install-recommends
# RUN rm -rf /var/lib/apt/lists/* && \
#     echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
#     apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN DEBIAN_FRONTEND=noninteractive apt update && \
#     DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-catkin ros-${ROS_DISTRO}-ros-base ros-${ROS_DISTRO}-vision-msgs ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-rosbridge-server python3-catkin-tools python3-venv python3-pip python3-rosdep python3-rosinstall python-is-python3 python3-rospy --no-install-recommends && \
#     rosdep init && rosdep update
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# # Install tmux and TMuLE
# RUN DEBIAN_FRONTEND=noninteractive pip install tmule

# # Clone TIAGo dependencies
# # RUN mkdir /home/lcastor
# RUN cd /
# RUN wget https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/${ROS_DISTRO}-devel/tiago_public-${ROS_DISTRO}.rosinstall
# RUN rosinstall /ros_ws/src /opt/ros/${ROS_DISTRO} tiago_public-${ROS_DISTRO}.rosinstall
# RUN apt update; apt upgrade -y

# # install deps
# RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-rosbag \
#     ros-noetic-actionlib-msgs \
#     ros-noetic-rqt-gui-py \
#     ros-noetic-teb-local-planner \
#     ros-noetic-nodelet \
#     ros-noetic-sensor-msgs \
#     ros-noetic-message-generation \
#     ros-noetic-std-msgs \
#     ros-noetic-visualization-msgs \
#     ros-noetic-joy \
#     ros-noetic-std-srvs \
#     ros-noetic-rqt-gui \
#     ros-noetic-twist-mux \
#     ros-noetic-map-server \
#     ros-noetic-cmake-modules \
#     ros-noetic-nav-msgs \
#     ros-noetic-control-toolbox \
#     ros-noetic-xacro \
#     ros-noetic-urdf-geometry-parser \
#     ros-noetic-realtime-tools \
#     ros-noetic-geometry-msgs \
#     ros-noetic-tf2-eigen \
#     ros-noetic-image-transport \
#     ros-noetic-gazebo-ros-pkgs \
#     ros-noetic-gazebo-ros-control \
#     ros-noetic-tf-conversions \
#     ros-noetic-four-wheel-steering-msgs \
#     ros-build-essential
# RUN apt update; apt upgrade -y

# setting up user
ARG UNAME=plushri
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN echo 'plushri:plushri' | chpasswd
RUN adduser $UNAME sudo
# WORKDIR /home/plushri/ros_ws
# RUN cp -r /ros_ws /home/plushri
# RUN chown -R $UNAME:$UNAME /home/plushri
# RUN chmod 755 /home/plushri
# RUN echo "source /home/plushri/ros_ws/devel/setup.bash" >> /home/plushri/.bashrc
# USER $UNAME


# # build ros ws
# RUN cd /home/plushri/ros_ws; . /opt/ros/${ROS_DISTRO}/setup.sh; catkin build
# USER plushri

# Fix pip not working properly
# RUN wget https://files.pythonhosted.org/packages/00/3f/ea5cfb789dddb327e6d2cf9377c36d9d8607af85530af0e7001165587ae7/pyOpenSSL-22.1.0-py3-none-any.whl
# RUN python3 -m easy_install pyOpenSSL-22.1.0-py3-none-any.whl
# #

# Set environment variables
ENV ROS_LOCALHOST_ONLY=0
ENV LIBGL_ALWAYS_SOFTWARE=1

# Create entrypoint for image
CMD ["/bin/bash"]