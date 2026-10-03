#include <rclcpp/rclcpp.hpp>
#include <fishbone/vslam/data_loader.hpp>
#include <iostream>
#include <iomanip>
#include <cstdio>

using namespace FISHBONE::VSLAM;

class VSLAMInspector : public rclcpp::Node {
public:
    VSLAMInspector() : Node("vslam_inspector"), frame_idx_(0) {
        // Load calibration and ground truth
        std::string calib_file = "/workspace/data/dataset/sequences/00/calib.txt";
        std::string poses_file = "/workspace/data/dataset/poses/00.txt";
        
        if (!DataLoader::loadCalibration(calib_file, calib_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration");
            return;
        }
        
        if (!DataLoader::loadGroundTruthPoses(poses_file, gt_poses_)) {
            RCLCPP_WARN(this->get_logger(), "Failed to load ground truth poses");
        }
        
        // Print header
        printHeader();
        
        // Create timer for frame updates
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&VSLAMInspector::displayFrame, this));
    }
    
private:
    CalibrationData calib_;
    std::vector<GroundTruthPose> gt_poses_;
    rclcpp::TimerBase::SharedPtr timer_;
    int frame_idx_;
    
    void printHeader() {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              VSLAM Frame Inspector - Ground Truth Viewer                   ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
        
        std::cout << "Dataset Information:\n";
        std::cout << "  Location: /workspace/data/dataset/sequences/00/\n";
        std::cout << "  Total Frames: 4541\n";
        std::cout << "  Image Size: 1241 × 376\n";
        std::cout << "  Baseline: " << std::fixed << std::setprecision(3) << calib_.baseline << " m\n";
        std::cout << "  Focal Length: " << std::fixed << std::setprecision(1) << calib_.K(0, 0) << " px\n";
        std::cout << "\n";
    }
    
    void displayFrame() {
        if (frame_idx_ >= (int)gt_poses_.size()) {
            frame_idx_ = 0;
        }
        
        Eigen::Matrix4d pose = gt_poses_[frame_idx_].T_w_c;
        
        // Extract position and rotation
        Eigen::Vector3d position = pose.topRightCorner(3, 1);
        Eigen::Matrix3d rotation = pose.topLeftCorner(3, 3);
        
        // Compute rotation angles (approximation)
        double roll = std::atan2(rotation(2, 1), rotation(2, 2));
        double pitch = std::atan2(-rotation(2, 0), std::sqrt(rotation(2, 1)*rotation(2, 1) + rotation(2, 2)*rotation(2, 2)));
        double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
        
        // Display frame info
        std::cout << "\r";
        std::cout << "Frame: " << std::setw(5) << std::left << frame_idx_ << " | ";
        std::cout << "Pos: (" << std::fixed << std::setprecision(2) 
                  << std::setw(7) << position(0) << ", "
                  << std::setw(7) << position(1) << ", "
                  << std::setw(7) << position(2) << ") m | ";
        std::cout << "Rot: (" << std::setw(6) << (roll*180/M_PI) << "°, "
                  << std::setw(6) << (pitch*180/M_PI) << "°, "
                  << std::setw(6) << (yaw*180/M_PI) << "°) ";
        
        std::cout.flush();
        
        frame_idx_++;
        if (frame_idx_ >= 100) {  // Show first 100 frames
            std::cout << "\n\n✅ Ground truth inspection complete!\n";
            std::cout << "Total frames displayed: " << frame_idx_ << "\n\n";
            rclcpp::shutdown();
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VSLAMInspector>());
    rclcpp::shutdown();
    return 0;
}
