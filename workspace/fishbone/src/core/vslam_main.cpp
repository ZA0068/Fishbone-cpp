#include <rclcpp/rclcpp.hpp>
#include <fishbone/vslam/tracker.hpp>
#include <fishbone/vslam/data_loader.hpp>
#include <fishbone/vslam/evaluator.hpp>
#include <fishbone/vslam/gpu_detector.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <cstdio>

using namespace FISHBONE::VSLAM;

class VSLAMNode : public rclcpp::Node {
public:
    VSLAMNode() : Node("vslam_node") {
        // Initialize GPU detection
        auto& gpu_detector = GPUDetector::getInstance();
        gpu_detector.initialize();
        
        bool use_gpu = gpu_detector.hasGPU();
        RCLCPP_INFO(this->get_logger(), 
                    "VSLAM Node started - GPU: %s, Backend: %s",
                    use_gpu ? "YES" : "NO", 
                    gpu_detector.backendName().c_str());
        
        tracker_ = std::make_unique<Tracker>(use_gpu);
        
        // Load calibration
        std::string calib_file = "/workspace/data/dataset/sequences/00/calib.txt";
        if (!DataLoader::loadCalibration(calib_file, calib_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Calibration loaded successfully");
        
        // Load ground truth poses
        std::string poses_file = "/workspace/data/dataset/poses/00.txt";
        if (!DataLoader::loadGroundTruthPoses(poses_file, gt_poses_)) {
            RCLCPP_WARN(this->get_logger(), "Failed to load ground truth poses");
        }
        
        tracker_->initialize(cv::Mat());  // Initialize tracker
        
        // Create a timer to process frames from dataset
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&VSLAMNode::processFrame, this));
        
        frame_idx_ = 0;
    }
    
private:
    std::unique_ptr<Tracker> tracker_;
    CalibrationData calib_;
    std::vector<GroundTruthPose> gt_poses_;
    std::vector<PoseEstimate> estimated_poses_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    int frame_idx_;
    
    void processFrame() {
        // Load images from dataset
        std::string img_left_file = std::string("/workspace/data/dataset/sequences/00/image_2/") +
                                   std::to_string(frame_idx_).substr(0, 6) + ".png";
        std::string img_right_file = std::string("/workspace/data/dataset/sequences/00/image_3/") +
                                    std::to_string(frame_idx_).substr(0, 6) + ".png";
        
        // Format frame index with leading zeros
        char buffer[7];
        snprintf(buffer, sizeof(buffer), "%06d", frame_idx_);
        img_left_file = std::string("/workspace/data/dataset/sequences/00/image_2/") + buffer + ".png";
        img_right_file = std::string("/workspace/data/dataset/sequences/00/image_3/") + buffer + ".png";
        
        // Load velodyne points
        std::string lidar_file = std::string("/workspace/data/dataset/sequences/00/velodyne/") + buffer + ".bin";
        
        cv::Mat left_img = cv::imread(img_left_file, cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(img_right_file, cv::IMREAD_GRAYSCALE);
        
        if (left_img.empty() || right_img.empty()) {
            RCLCPP_WARN(this->get_logger(), "Cannot load images for frame %d", frame_idx_);
            frame_idx_++;
            if (frame_idx_ >= 4541) {
                finalizeAndExit();
            }
            return;
        }
        
        // Load velodyne points
        std::vector<cv::Point3f> lidar_points = loadVelodynePoints(lidar_file);
        
        // Track frame
        Frame frame;
        if (tracker_->track(left_img, right_img, lidar_points, frame)) {
            PoseEstimate est;
            est.frame_id = frame_idx_;
            est.T_w_c = tracker_->getPose();
            estimated_poses_.push_back(est);
            
            RCLCPP_INFO(this->get_logger(), 
                        "Frame %d tracked. Pose: [%.3f, %.3f, %.3f]",
                        frame_idx_,
                        est.T_w_c(0, 3), est.T_w_c(1, 3), est.T_w_c(2, 3));
        }
        
        frame_idx_++;
        
        // Process only first 100 frames for testing
        if (frame_idx_ >= 100) {
            finalizeAndExit();
        }
    }
    
    std::vector<cv::Point3f> loadVelodynePoints(const std::string& filename) {
        std::vector<cv::Point3f> points;
        std::ifstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            return points;
        }
        
        float x, y, z, intensity;
        while (file.read(reinterpret_cast<char*>(&x), 4) &&
               file.read(reinterpret_cast<char*>(&y), 4) &&
               file.read(reinterpret_cast<char*>(&z), 4) &&
               file.read(reinterpret_cast<char*>(&intensity), 4)) {
            points.emplace_back(x, y, z);
        }
        file.close();
        return points;
    }
    
    void finalizeAndExit() {
        RCLCPP_INFO(this->get_logger(), "Processing complete. Evaluating...");
        
        // Evaluate
        if (estimated_poses_.size() > 0 && gt_poses_.size() > 0) {
            std::vector<PoseEstimate> gt_est;
            for (size_t i = 0; i < std::min(estimated_poses_.size(), gt_poses_.size()); i++) {
                PoseEstimate pe;
                pe.frame_id = i;
                pe.T_w_c = gt_poses_[i].T_w_c;
                gt_est.push_back(pe);
            }
            
            EvaluationMetrics metrics = Evaluator::evaluate(estimated_poses_, gt_est);
            
            RCLCPP_INFO(this->get_logger(), 
                        "=== Evaluation Results ===");
            RCLCPP_INFO(this->get_logger(), 
                        "ATE (Absolute Trajectory Error): %.4f ± %.4f m",
                        metrics.ate_mean, metrics.ate_std);
            RCLCPP_INFO(this->get_logger(), 
                        "RPE (Relative Pose Error): %.4f ± %.4f m",
                        metrics.rpe_mean, metrics.rpe_std);
            
            // Generate trajectory plot
            Evaluator::plotTrajectory(estimated_poses_, gt_est, "trajectory.png");
        }
        
        rclcpp::shutdown();
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VSLAMNode>());
    rclcpp::shutdown();
    return 0;
}
