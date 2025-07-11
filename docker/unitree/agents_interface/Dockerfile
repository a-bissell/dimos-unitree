FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Set ROS distro
ENV ROS_DISTRO=humble

# Install basic requirements
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    clang \
    portaudio19-dev \
    git \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    software-properties-common \
    libxcb1-dev \
    libxcb-keysyms1-dev \
    libxcb-util0-dev \
    libxcb-icccm4-dev \
    libxcb-image0-dev \
    libxcb-randr0-dev \
    libxcb-shape0-dev \
    libxcb-xinerama0-dev \
    libxcb-xkb-dev \
    libxkbcommon-x11-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    supervisor \
    screen \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install specific numpy version first
RUN pip install 'numpy<2.0.0'

# Add ROS2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 packages and dependencies
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-image-tools \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    ros-${ROS_DISTRO}-twist-mux \
    ros-${ROS_DISTRO}-joy \
    ros-${ROS_DISTRO}-teleop-twist-joy \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-nav2-amcl \
    ros-${ROS_DISTRO}-nav2-map-server \
    ros-${ROS_DISTRO}-nav2-util \
    ros-${ROS_DISTRO}-pointcloud-to-laserscan \
    ros-${ROS_DISTRO}-slam-toolbox \
    ros-${ROS_DISTRO}-foxglove-bridge \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-colcon-common-extensions \
    python3-vcstool \
    build-essential \
    screen \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Create workspace
WORKDIR /ros2_ws

# Clone the repository with submodules
RUN git clone --branch prod --recurse-submodules https://github.com/dimensionalOS/go2_ros2_sdk src

# Install Python requirements 
RUN cd src && pip install -r requirements.txt

# Create dimos directory structure
RUN mkdir -p /app/dimos /app/docker

COPY requirements.txt /app/

COPY base-requirements.txt /app/

WORKDIR /app

# Copy git configuration files first to initialize submodules
COPY .gitmodules ./
COPY .git ./.git

# Initialize and update submodules
RUN git submodule init && git submodule update --recursive

# Install torch and torchvision first due to builds in requirements.txt
RUN pip install --no-cache-dir -r base-requirements.txt

# Install dimos requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH permanently
ENV PYTHONPATH=/app:${PYTHONPATH}

# Install ROS dependencies
WORKDIR /ros2_ws
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build

# Source ROS2 and workspace in bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

COPY docker /app/docker/

# Setup supervisor configuration
COPY docker/unitree/agents_interface/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY docker/unitree/agents_interface/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy dimos and tests after submodules are initialized
COPY dimos /app/dimos/
COPY tests /app/tests
COPY dimos/__init__.py /app/__init__.py

# Change working directory to /app for proper relative pathing
WORKDIR /app

# Create output directories for supervisord and ROS
RUN mkdir -p /app/assets/output/
RUN mkdir -p /app/assets/output/ros

# TODO: Cleanup multiple working directories and seprate the dockerfiles for each service.

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
