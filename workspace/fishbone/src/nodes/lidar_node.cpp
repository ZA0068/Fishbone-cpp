#include <fishbone/nodes/lidar_node.hpp>

namespace FISHBONE {

LidarNode::LidarNode() : Node("lidar_node"), lidar_index_(0)
{
    // 1. Initialize Publisher with Custom Message
    lidar_publisher_ = this->create_publisher<fishbone::msg::LidarData>(
        "/fishbone/lidar", 10);
    
    // 2. Load File Paths
    // Defaulting to sequence 00, you can make this a parameter later
    auto data_path = this->LidarFilesPath("00");
    
    if (std::filesystem::exists(data_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(data_path)) {
            if (entry.path().extension() == ".bin") {
                file_list_.push_back(entry.path().string());
            }
        }
        std::sort(file_list_.begin(), file_list_.end());
        RCLCPP_INFO(this->get_logger(), "Found %zu velodyne files in %s", 
                    file_list_.size(), data_path.c_str());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Data path not found: %s", data_path.c_str());
    }
    
    // 3. Start Timer (10Hz = 100ms)
    if (!file_list_.empty()) {
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&LidarNode::timer_callback_static, this));
    }
    
    RCLCPP_INFO(this->get_logger(), "Lidar Node started.");
}

void LidarNode::timer_callback_static()
{
    if (file_list_.empty()) return;

    // Stop once the final scan has been published -- don't loop back to scan 0.
    if (lidar_index_ >= file_list_.size()) {
        RCLCPP_INFO(this->get_logger(), "Reached end of sequence (%zu scans published), stopping.",
            file_list_.size());
        timer_->cancel();
        return;
    }
    
    // 1. Read Binary Data
    std::vector<fishbone::msg::LidarPoint> points = read_velodyne_file(file_list_[lidar_index_]);
    
    // 2. Create Custom Message
    fishbone::msg::LidarData msg;
    msg.timestamp = this->now().seconds();
    msg.num_points = static_cast<uint32_t>(points.size());
    msg.points = points; // std::vector copy happens here

    // 3. Publish
    lidar_publisher_->publish(msg);
    
    RCLCPP_INFO_ONCE(this->get_logger(), "Publishing Lidar Data...");
    
    lidar_index_++;
}

std::vector<fishbone::msg::LidarPoint> LidarNode::read_velodyne_file(const std::string& filename)
{
    std::vector<fishbone::msg::LidarPoint> points;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        RCLCPP_WARN(this->get_logger(), "Could not open lidar file: %s", filename.c_str());
        return points;
    }
    
    // Determine file size to reserve memory
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate number of points (each point is 4 floats: x, y, z, intensity)
    // 4 bytes * 4 values = 16 bytes per point
    size_t num_points = size / (4 * sizeof(float));
    points.reserve(num_points);

    // Buffer to read raw floats
    std::vector<float> buffer(4); 

    // Read loop
    // KITTI .bin format: [x, y, z, intensity] as float32
    while (file.read(reinterpret_cast<char*>(buffer.data()), 4 * sizeof(float))) {
        fishbone::msg::LidarPoint p;
        p.x = buffer[0];
        p.y = buffer[1];
        p.z = buffer[2];
        p.intensity = buffer[3];
        
        // KITTI raw binary does not have ring or time info, setting defaults
        p.ring = 0;
        p.time = 0;

        points.push_back(p);
    }
    
    file.close();
    return points;
}

} // namespace FISHBONE