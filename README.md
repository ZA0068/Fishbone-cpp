# VSLAM SOFT2 Development Environment

This project provides a complete development environment for OpenCV VSLAM SOFT2 using Podman containers with GPU support.

## 🚀 Quick Start

### Prerequisites

1. **Podman** - Container runtime
2. **podman-compose** - Docker Compose equivalent for Podman
3. **NVIDIA Container Toolkit** - For GPU support

```bash
# Install podman-compose
pip install podman-compose

# Install NVIDIA Container Toolkit (if not already installed)
# Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Getting Started

1. **Start the development environment:**
   ```bash
   ./dev-env.sh start
   ```

2. **Enter the development container:**
   ```bash
   ./dev-env.sh shell
   ```

3. **Build the SOFT2 project:**
   ```bash
   ./dev-env.sh build
   ```

4. **Run tests:**
   ```bash
   ./dev-env.sh test
   ```

## 📦 Container Contents

The development container (`zainahmed1997/opencv_vslam:tagname`) includes:

- **NVIDIA CUDA 12.8** - Latest GPU computing platform
- **OpenCV** - Computer vision library
- **Pangolin** - 3D visualization library
- **G2O** - Graph optimization library
- **Eigen** - Linear algebra library
- **ROS2 Jazzy** - Robot Operating System 2
- **Anaconda** - Python environment with Python 3.12
- **Ubuntu 24.04** - Base operating system

## 🛠 Development Commands

| Command | Description |
|---------|-------------|
| `./dev-env.sh start` | Start the development environment |
| `./dev-env.sh stop` | Stop the development environment |
| `./dev-env.sh shell` | Enter the container shell |
| `./dev-env.sh logs` | View container logs |
| `./dev-env.sh build` | Build the SOFT2 project |
| `./dev-env.sh test` | Run SOFT2 tests |
| `./dev-env.sh status` | Check environment status |
| `./dev-env.sh help` | Show help information |

## 📁 Project Structure

```
/home/zain/Projects/VSLAM/
├── SOFT2.h                 # SOFT2 class header
├── SOFT2.cpp               # SOFT2 class implementation
├── docker-compose.yml      # Podman compose configuration
├── dev-env.sh              # Development environment manager
├── README.md               # This file
└── papers/                 # Research papers (PDF files)
```

## 🔧 Configuration Details

### Volume Mounts

- **Project directory**: `./` → `/workspace` (Your code)
- **X11 forwarding**: `/tmp/.X11-unix` (GUI support)
- **Camera devices**: `/dev/video*` (Camera access)
- **GPU devices**: `/dev/dri` (GPU access)

### Environment Variables

- `DISPLAY`: X11 display forwarding
- `CUDA_VISIBLE_DEVICES=all`: GPU access
- `PYTHONPATH`: Python module search path
- `ROS_DISTRO=jazzy`: ROS2 distribution
- `WORKSPACE_DIR=/workspace`: Working directory

### GPU Support

The container is configured with full NVIDIA GPU support:
- CUDA 12.8 runtime
- All GPU capabilities enabled
- Proper device mounting

## 🐍 Python Environment

- **Native Python**: 3.12
- **Anaconda**: Full scientific Python stack
- **OpenCV Python bindings**: Available
- **Custom packages**: Install with `pip` or `conda`

## 🤖 ROS2 Integration

ROS2 Jazzy is pre-installed and configured:
- Domain ID: 42
- All standard ROS2 tools available
- Ready for SLAM node development

## 🔍 SOFT2 Class

The SOFT2 class provides:
- Data management and processing
- VSLAM-ready structure
- OpenCV integration points
- Modular design for extensions

### Example Usage

```cpp
#include "SOFT2.h"

int main() {
    SOFT2 slam_system("VSLAM_SOFT2", 1);
    
    // Add some data points
    slam_system.addData(1.5);
    slam_system.addData(2.3);
    slam_system.addData(0.8);
    
    // Process and display
    slam_system.processData();
    slam_system.displayInfo();
    
    return 0;
}
```

## 🚨 Troubleshooting

### GPU Issues
- Ensure NVIDIA drivers are installed on host
- Check NVIDIA Container Toolkit installation
- Verify GPU visibility: `nvidia-smi`

### X11 Issues
- Run `xhost +local:root` on host
- Check DISPLAY variable: `echo $DISPLAY`
- Ensure X11 forwarding is enabled

### Container Issues
- Check container status: `./dev-env.sh status`
- View logs: `./dev-env.sh logs`
- Restart environment: `./dev-env.sh stop && ./dev-env.sh start`

## 📚 Development Workflow

1. **Start environment**: `./dev-env.sh start`
2. **Enter container**: `./dev-env.sh shell`
3. **Edit code**: Use VS Code or your preferred editor
4. **Build project**: `./dev-env.sh build`
5. **Test changes**: `./dev-env.sh test`
6. **Debug**: Use GDB, printf, or ROS2 tools
7. **Iterate**: Repeat steps 3-6

## 🎯 Next Steps

- Implement SLAM algorithms in SOFT2 class
- Add OpenCV image processing methods
- Integrate with ROS2 for sensor data
- Develop Pangolin visualization
- Add G2O optimization routines

## 📖 References

- OpenCV Documentation: https://docs.opencv.org/
- ROS2 Documentation: https://docs.ros.org/en/jazzy/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- Pangolin Documentation: https://github.com/stevenlovegrove/Pangolin

---

Happy SLAM development! 🚀