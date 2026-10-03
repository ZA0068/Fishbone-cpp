#ifndef FISHBONE_NODES_VIDEOSTREAM_NODE_HPP_  // Fixed Include Guard
#define FISHBONE_NODES_VIDEOSTREAM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <fishbone/msg/frame.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp> // Using .hpp as requested
#include <opencv2/opencv.hpp>

namespace FISHBONE
{

    class VideoStreamNode : public rclcpp::Node
    {
    public:
        VideoStreamNode();
        
    private:
        // Member variable for subscription
        rclcpp::Subscription<fishbone::msg::Frame>::SharedPtr subscription_;
        
        // Callback function declaration
        void image_callback(const fishbone::msg::Frame::SharedPtr msg);
    };
}

#endif