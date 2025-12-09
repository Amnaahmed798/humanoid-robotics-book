# NVIDIA Isaac SDK and Isaac Sim Setup

## Introduction

The NVIDIA Isaac platform provides a comprehensive set of tools for developing AI-powered robots. This chapter covers the installation and setup of the Isaac SDK and Isaac Sim, which are essential for leveraging NVIDIA's AI capabilities in robotics applications.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with Compute Capability 6.0 or higher (Pascal architecture or newer)
  - Recommended: RTX 3080, RTX 4090, A40, A100, or better
- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ available space
- **OS**: Ubuntu 20.04 LTS or 22.04 LTS (recommended)

### Software Requirements
- **CUDA**: Version 11.8 or later
- **Docker**: Version 20.10 or later with nvidia-docker2
- **NVIDIA Driver**: Version 520 or later
- **ROS 2**: Humble Hawksbill (recommended)

## Installing NVIDIA Drivers

### Check GPU Information
```bash
lspci | grep -i nvidia
```

### Install NVIDIA Drivers
```bash
# Update package list
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific driver version
sudo apt install nvidia-driver-535

# Reboot system
sudo reboot
```

### Verify GPU Installation
```bash
nvidia-smi
```

## Installing CUDA

### Download and Install CUDA
```bash
# Download CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run

# Run installer (do NOT install driver, only CUDA toolkit)
sudo sh cuda_12.3.0_545.23.06_linux.run
```

During installation:
- Uncheck "Driver" option (keep existing driver)
- Check "CUDA Toolkit" and "CUDA Samples"
- Accept default installation path: `/usr/local/cuda-12.3`

### Set Environment Variables
Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Apply changes:
```bash
source ~/.bashrc
```

### Verify CUDA Installation
```bash
nvcc --version
nvidia-ml-py3
```

## Installing Docker and nvidia-docker

### Install Docker
```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

# Add Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### Install nvidia-docker
```bash
# Add NVIDIA package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package list
sudo apt update

# Install nvidia-docker2
sudo apt install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

### Test GPU in Docker
```bash
sudo docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

## Installing Isaac Sim

### Option 1: Using Isaac Sim Omniverse Launcher (Recommended)

1. Download Isaac Sim from NVIDIA Developer website
2. Install Omniverse Launcher
3. Launch Isaac Sim through the launcher

### Option 2: Using Docker (Alternative)

```bash
# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# Run Isaac Sim container
docker run --gpus all -it --rm \
  --network=host \
  --env "ACCEPT_EULA=Y" \
  --env "PRIVACY_CONSENT=Y" \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

### Option 3: Local Installation

1. Download Isaac Sim from NVIDIA Developer website
2. Extract the archive
3. Run the installation script

## Installing Isaac ROS

### Using Debian Packages (Recommended)
```bash
# Add NVIDIA package repositories
curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -sL https://repo.download.nvidia.com/jetson-agx-xavier-silicon/76b89119c29c495dcb155d45365211a18a0816b6/nvidia.jetson-agx-xavier-silicon.asc | sudo apt-key add -

# Add Isaac ROS repository
sudo sh -c 'echo "deb https://packages.nvidia.com/repos/ubuntu2204/isaac_ros main" > /etc/apt/sources.list.d/nvidia-isaac-ros.list'

# Update package list
sudo apt update

# Install Isaac ROS packages
sudo apt install -y \
  ros-humble-isaac-ros-apriltag \
  ros-humble-isaac-ros-argus-camera \
  ros-humble-isaac-ros-bitmask-msgs \
  ros-humble-isaac-ros-bridge-common \
  ros-humble-isaac-ros-bytetrack \
  ros-humble-isaac-ros-cask-parser \
  ros-humble-isaac-ros-cask-transport \
  ros-humble-isaac-ros-dapple-pose \
  ros-humble-isaac-ros-diff-odometry \
  ros-humble-isaac-ros-dope \
  ros-humble-isaac-ros-ego-tracker \
  ros-humble-isaac-ros-elevator \
  ros-humble-isaac-ros-face-detector \
  ros-humble-isaac-ros-fiducial-pose-estimator \
  ros-humble-isaac-ros-gscam \
  ros-humble-isaac-ros-gxf \
  ros-humble-isaac-ros-h264-encoder \
  ros-humble-isaac-ros-h264-decoder \
  ros-humble-isaac-ros-hawk \
  ros-humble-isaac-ros-image-ros \
  ros-humble-isaac-ros-image-transport \
  ros-humble-isaac-ros-isaac-ros-messages \
  ros-humble-isaac-ros-isaac-ros-operators \
  ros-humble-isaac-ros-isaac-ros-tests \
  ros-humble-isaac-ros-isaac-ros-utils \
  ros-humble-isaac-ros-lidar-ros \
  ros-humble-isaac-ros-localization \
  ros-humble-isaac-ros-mi-ros-bridge \
  ros-humble-isaac-ros-multipart-publisher \
  ros-humble-isaac-ros-nitros-camera-calibrator \
  ros-humble-isaac-ros-nitros-image-publisher \
  ros-humble-isaac-ros-nitros-image-rectification \
  ros-humble-isaac-ros-nitros-image-ros \
  ros-humble-isaac-ros-nitros-point-cloud-publisher \
  ros-humble-isaac-ros-nitros-point-cloud-ros \
  ros-humble-isaac-ros-novatel-gps \
  ros-humble-isaac-ros-object-analytics \
  ros-humble-isaac-ros-occupancy-grid-localizer \
  ros-humble-isaac-ros-omniverse-isaac-sim \
  ros-humble-isaac-ros-people-segmentation \
  ros-humble-isaac-ros-py-test \
  ros-humble-isaac-ros-realsense \
  ros-humble-isaac-ros-se3 \
  ros-humble-isaac-ros-segment-any-thing \
  ros-humble-isaac-ros-stereo-image-publisher \
  ros-humble-isaac-ros-stereo-rectification \
  ros-humble-isaac-ros-stereo-tilt \
  ros-humble-isaac-ros-test \
  ros-humble-isaac-ros-test-image \
  ros-humble-isaac-ros-test-point-cloud \
  ros-humble-isaac-ros-test-se3 \
  ros-humble-isaac-ros-test-transport \
  ros-humble-isaac-ros-test-utils \
  ros-humble-isaac-ros-thermal-image-ros \
  ros-humble-isaac-ros-thermal-publisher \
  ros-humble-isaac-ros-thermal-utils \
  ros-humble-isaac-ros-visual-odometry \
  ros-humble-isaac-ros-wheel-odometry
```

### Building from Source (Alternative)

```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Clone Isaac ROS repositories
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git src/isaac_ros_apriltag
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pointcloud_utils.git src/isaac_ros_pointcloud_utils

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build --symlink-install
source install/setup.bash
```

## Setting up Isaac Sim Environment

### Configuring Isaac Sim for ROS 2

Isaac Sim can interface with ROS 2 through extensions:

1. Launch Isaac Sim
2. Go to Window → Extensions
3. Search for and enable ROS 2 Bridge extension
4. Configure ROS domain ID to match your ROS 2 setup

### Isaac Sim Extensions for Robotics

Common extensions for robotics development:
- **ROS2 Bridge**: Enables ROS 2 communication
- **Isaac Sim Python**: Provides Python scripting capabilities
- **Isaac Sim Sensors**: Advanced sensor simulation
- **Isaac Sim Navigation**: Navigation and path planning tools

## Verification and Testing

### Test Isaac ROS Installation
```bash
# Source ROS 2 and Isaac ROS
source /opt/ros/humble/setup.bash
source ~/isaac_ros_ws/install/setup.bash  # if built from source

# Check available Isaac ROS packages
ros2 pkg list | grep isaac
```

### Test Isaac Sim (if installed locally)
```bash
# Navigate to Isaac Sim directory
cd ~/isaac-sim-4.2.0

# Run Isaac Sim
./isaac-sim.sh
```

### Run Isaac ROS Demo
```bash
# Source ROS 2 and Isaac ROS
source /opt/ros/humble/setup.bash
source ~/isaac_ros_ws/install/setup.bash

# Run Visual SLAM demo (if available)
ros2 launch isaac_ros_visual_slam visual_slam_node.launch.py
```

## Troubleshooting Common Issues

### GPU Access Issues
- Verify NVIDIA driver installation: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Test Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi`

### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Isaac Sim Launch Issues
- Check Omniverse logs in `~/.nvidia-omniverse/logs/`
- Verify GPU compatibility
- Check system requirements

### Isaac ROS Package Issues
- Verify ROS 2 humble installation
- Check for missing dependencies: `rosdep check --from-paths src --ignore-src`
- Rebuild workspace if needed: `colcon build --symlink-install`

## Best Practices

1. **Keep NVIDIA drivers updated**: Regularly update to latest stable drivers
2. **Use Docker when possible**: Easier to manage dependencies and versions
3. **Check compatibility**: Ensure Isaac Sim, Isaac ROS, and ROS 2 versions are compatible
4. **Monitor resources**: Isaac Sim can be resource-intensive
5. **Save configurations**: Document your working setup for reproducibility
6. **Use virtual environments**: Isolate Isaac development from other projects

With the Isaac SDK and Isaac Sim properly installed, you're ready to explore the advanced AI capabilities for robotics in the following chapters.