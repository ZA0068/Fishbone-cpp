# 1. Base Image
FROM docker.io/zainahmed1997/opencv_vslam:1.2.3

# 2. Switch to ROOT to allow installation
USER root
ENV DEBIAN_FRONTEND=noninteractive

# --- ROS 2 JAZZY INSTALLATION ---

# A. Setup Repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    && add-apt-repository universe

# B. Add GPG Key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# C. Add ROS 2 Repository
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# D. Install ROS 2 JAZZY + Fixes
# Added 'python3-yaml' to fix the "ModuleNotFoundError: No module named 'yaml'" crash
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-yaml \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# E. Initialize rosdep
RUN rosdep init || true && rosdep update

# --- END ROS 2 INSTALLATION ---

# 3. Install Linker Fixes (GDAL/CURL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libgdal-dev \
    libopenjp2-7-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 4. Setup Environment
# Ensure ROS is sourced automatically for every new shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

# 5. Switch back to non-root user
USER 1000
WORKDIR /workspace