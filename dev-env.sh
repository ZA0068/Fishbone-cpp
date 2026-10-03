#!/bin/bash

# VSLAM SOFT3 Development Environment Manager
# This script helps manage the Podman compose environment for OpenCV VSLAM development with IMU/RTK/LiDAR support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} VSLAM SOFT3 Development Environment${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if podman-compose is available
check_dependencies() {
    if ! command -v podman-compose &> /dev/null; then
        print_error "podman-compose is not installed. Please install it first:"
        echo "  pip install podman-compose"
        exit 1
    fi
    
    if ! command -v podman &> /dev/null; then
        print_error "podman is not installed. Please install it first."
        exit 1
    fi
}

# Start the development environment
start_env() {
    print_header
    print_status "Starting VSLAM SOFT3 development environment..."
    
    # Set DISPLAY for X11 forwarding
    export DISPLAY=${DISPLAY:-:0}
    
    # Allow X11 connections (for GUI applications)
    xhost +local:root 2>/dev/null || print_warning "Could not configure X11 forwarding"
    
    podman-compose up -d
    
    print_status "Environment started successfully!"
    print_status "Container name: vslam_soft3_development"
    echo ""
    print_status "To enter the development environment, run:"
    echo "  ./dev-env.sh shell"
    echo ""
    print_status "To view logs, run:"
    echo "  ./dev-env.sh logs"
}

# Stop the development environment
stop_env() {
    print_status "Stopping VSLAM SOFT3 development environment..."
    podman-compose down
    print_status "Environment stopped successfully!"
}

# Enter the container shell
enter_shell() {
    print_status "Entering VSLAM SOFT3 development container..."
    podman exec -it vslam_soft3_development bash
}

# Show container logs
show_logs() {
    podman-compose logs -f vslam-soft3-dev
}

# Build and compile SOFT3 project using CMake
build_project() {
    print_status "Building SOFT3 project inside container..."
    podman exec -it vslam_soft3_development bash -c "
        cd /workspace || exit 1
        
        # Ensure dependencies are available
        echo '[INFO] Checking/installing dependencies...'
        apt-get update > /dev/null 2>&1 || true
        apt-get install -y libeigen3-dev cmake build-essential > /dev/null 2>&1 || true
        
        # Clean old CMake cache from host
        echo '[INFO] Cleaning old CMake cache...'
        rm -rf build/CMakeCache.txt build/CMakeFiles build/cmake_install.cmake
        
        # Create build directory
        mkdir -p build
        cd build
        
        echo '[INFO] Running CMake configure...'
        cmake .. -DCMAKE_BUILD_TYPE=Release
        
        if [ \$? -ne 0 ]; then
            echo '[ERROR] CMake configuration failed!'
            exit 1
        fi
        
        echo '[INFO] Building with make...'
        make -j\$(nproc)
        
        if [ \$? -eq 0 ]; then
            echo '[✓] SOFT3 build successful!'
            echo '[INFO] Executable: ./soft3_main'
            ls -lh soft3_main 2>/dev/null || true
        else
            echo '[ERROR] Build failed!'
            exit 1
        fi
    "
}

# Run tests
run_tests() {
    print_status "Running SOFT3 executable with test data..."
    podman exec -it vslam_soft3_development bash -c "
        cd /workspace/build || exit 1
        
        if [ -f soft3_main ]; then
            echo '[✓] Running soft3_main...'
            ./soft3_main
            echo '[INFO] Test completed!'
        else
            echo '[ERROR] soft3_main not found. Run build first: ./dev-env.sh build'
            exit 1
        fi
    "
}

# Check environment status
check_status() {
    print_header
    print_status "Checking environment status..."
    
    if podman ps | grep -q vslam_soft3_development; then
        print_status "✅ Container is running"
        
        print_status "Container details:"
        podman exec vslam_soft3_development bash -c "
            echo 'CUDA Version:' && nvcc --version | head -1
            echo 'Python Version:' && python3 --version
            echo 'OpenCV Version:' && python3 -c 'import cv2; print(f\"OpenCV {cv2.__version__}\")'
            echo 'ROS2 Version:' && ros2 --version 2>/dev/null || echo 'ROS2 not available in PATH'
            echo 'Working Directory:' && pwd
            echo 'Available files:' && ls -la /workspace/ | head -10
        "
    else
        print_warning "❌ Container is not running"
        echo "Run './dev-env.sh start' to start the environment"
    fi
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start the development environment"
    echo "  stop      Stop the development environment"
    echo "  shell     Enter the container shell"
    echo "  logs      Show container logs"
    echo "  build     Build the SOFT3 project with CMake"
    echo "  test      Run SOFT3 executable"
    echo "  status    Check environment status"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start the environment"
    echo "  $0 shell     # Enter development shell"
    echo "  $0 build     # Compile SOFT3 with CMake"
    echo "  $0 test      # Run soft3_main tests"
}

# Main script logic
main() {
    check_dependencies
    
    case "${1:-}" in
        start)
            start_env
            ;;
        stop)
            stop_env
            ;;
        shell)
            enter_shell
            ;;
        logs)
            show_logs
            ;;
        build)
            build_project
            ;;
        test)
            run_tests
            ;;
        status)
            check_status
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"