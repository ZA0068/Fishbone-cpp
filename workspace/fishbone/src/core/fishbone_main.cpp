#include <rclcpp/rclcpp.hpp>
#include <fishbone/nodes/fishbone_node.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  // Create and spin the node
  rclcpp::spin(std::make_shared<FISHBONE::FishboneNode>());
  rclcpp::shutdown();
  return 0;
}