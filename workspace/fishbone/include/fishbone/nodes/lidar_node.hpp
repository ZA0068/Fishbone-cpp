#ifndef FISHBONE_NODES_LIDAR_NODE_HPP_
#define FISHBONE_NODES_LIDAR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

// Include your custom messages
#include <fishbone/msg/lidar_data.hpp>
#include <fishbone/msg/lidar_point.hpp>

namespace FISHBONE
{
    class LidarNode : public rclcpp::Node
    {
    public:
        LidarNode();
        
    private:
        rclcpp::Publisher<fishbone::msg::LidarData>::SharedPtr lidar_publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
        
        size_t lidar_index_;
        std::vector<std::string> file_list_; // Added to store paths

        void initializePointCloud();
        void timer_callback_static();
        
        // Helper to parse binary files into your Message type
        std::vector<fishbone::msg::LidarPoint> read_velodyne_file(const std::string& filename);

        std::string LidarFilesPath(const std::string &sequence)
        {
            return "/workspace/data/dataset/sequences/" + sequence + "/velodyne"; 
        }
    };
}

#endif // FISHBONE_NODES_LIDAR_NODE_HPP_