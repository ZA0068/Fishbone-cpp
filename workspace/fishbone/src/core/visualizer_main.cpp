#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <fishbone/vslam/data_loader.hpp>
#include <fstream>
#include <iostream>
#include <cstdio>

using namespace FISHBONE::VSLAM;

class VSLAMVisualizer : public rclcpp::Node {
public:
    VSLAMVisualizer() : Node("vslam_visualizer"), frame_idx_(0), paused_(false) {
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
        
        RCLCPP_INFO(this->get_logger(), "Visualizer initialized with %ld ground truth poses", gt_poses_.size());
        
        // Create timer for frame updates
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&VSLAMVisualizer::updateFrame, this));
    }
    
private:
    CalibrationData calib_;
    std::vector<GroundTruthPose> gt_poses_;
    rclcpp::TimerBase::SharedPtr timer_;
    int frame_idx_;
    bool paused_;
    
    void updateFrame() {
        if (paused_) return;
        
        // Load current frame images
        char buffer[7];
        snprintf(buffer, sizeof(buffer), "%06d", frame_idx_);
        
        std::string left_file = std::string("/workspace/data/dataset/sequences/00/image_2/") + buffer + ".png";
        std::string right_file = std::string("/workspace/data/dataset/sequences/00/image_3/") + buffer + ".png";
        
        cv::Mat left = cv::imread(left_file);
        cv::Mat right = cv::imread(right_file);
        
        if (left.empty() || right.empty()) {
            RCLCPP_WARN(this->get_logger(), "Cannot load frame %d", frame_idx_);
            frame_idx_++;
            if (frame_idx_ >= 4541) frame_idx_ = 0;
            return;
        }
        
        // Create display canvas
        cv::Mat display(left.rows, left.cols * 2 + 30, CV_8UC3, cv::Scalar(50, 50, 50));
        
        // Place images side by side
        left.copyTo(display(cv::Rect(0, 0, left.cols, left.rows)));
        right.copyTo(display(cv::Rect(left.cols + 30, 0, right.cols, right.rows)));
        
        // Add frame info
        std::string frame_text = "Frame: " + std::to_string(frame_idx_);
        cv::putText(display, frame_text, cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
        
        // Add ground truth pose if available
        if (frame_idx_ < (int)gt_poses_.size()) {
            Eigen::Matrix4d pose = gt_poses_[frame_idx_].T_w_c;
            std::string pose_text = cv::format("GT Pose: X=%.2f Y=%.2f Z=%.2f",
                                              pose(0, 3), pose(1, 3), pose(2, 3));
            cv::putText(display, pose_text, cv::Point(20, 100),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 1);
        }
        
        // Add controls info
        std::string controls = "Space: Play/Pause | N: Next | P: Previous | Q: Quit";
        cv::putText(display, controls, cv::Point(20, display.rows - 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        
        // Display
        cv::imshow("VSLAM Frame Viewer - Ground Truth Comparison", display);
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // Q or ESC
            rclcpp::shutdown();
            return;
        } else if (key == ' ') {  // Space
            paused_ = !paused_;
            RCLCPP_INFO(this->get_logger(), paused_ ? "Paused" : "Playing");
        } else if (key == 'n') {  // N (next)
            frame_idx_++;
            if (frame_idx_ >= 4541) frame_idx_ = 0;
            paused_ = true;
        } else if (key == 'p') {  // P (previous)
            frame_idx_--;
            if (frame_idx_ < 0) frame_idx_ = 4540;
            paused_ = true;
        }
        
        frame_idx_++;
        if (frame_idx_ >= 4541) frame_idx_ = 0;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VSLAMVisualizer>());
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}
