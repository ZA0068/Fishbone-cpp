#include <rclcpp/rclcpp.hpp>
#include <fishbone/nodes/videostream_node.hpp>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<FISHBONE::VideoStreamNode>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
