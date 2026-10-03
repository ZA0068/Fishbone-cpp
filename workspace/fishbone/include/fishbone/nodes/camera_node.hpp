#ifndef FISHBONE_NODES_CAMERA_NODE_HPP_
#define FISHBONE_NODES_CAMERA_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <utility>

// FIX 1: Include the correct message headers
#include <fishbone/msg/camera_data.hpp>
#include <fishbone/msg/stereo_camera_data.hpp>

namespace FISHBONE
{
    class CameraNode : public rclcpp::Node
    {
    public:
        CameraNode();
        
        std::pair<std::vector<std::string>, std::vector<std::string>> getImagePaths(const std::string &sequence);

    private:
        rclcpp::Publisher<fishbone::msg::CameraData>::SharedPtr mono_publisher_;
        rclcpp::Publisher<fishbone::msg::StereoCameraData>::SharedPtr stereo_publisher_;

        rclcpp::TimerBase::SharedPtr timer_;
        size_t camera_index_;
        std::vector<std::string> left_image_paths_;
        std::vector<std::string> right_image_paths_;

        bool is_stereo_mode_;
        bool is_color_mode_;

        // Calibration placeholders
        std::vector<double> p_left_, k_left_;
        std::vector<double> p_right_, k_right_;

        void initializeCamera(bool stream, bool is_stereo, bool is_color);
        void timer_callback_static();
        void loadImageFromFiles();

        std::pair<std::string, std::string> imagePaths(const std::string &sequence)
        {
            std::string base_path = "/workspace/data/dataset/sequences/" + sequence;
            // FIX 2: Use image_2 and image_3 for color data (KITTI standard)
            return {base_path + "/image_2/", base_path + "/image_3/"}; 
        }
    };
}

#endif // FISHBONE_NODES_CAMERA_NODE_HPP_