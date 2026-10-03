#include <rclcpp/rclcpp.hpp>
#include <fishbone/nodes/lidar_node.hpp>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<FISHBONE::LidarNode>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
