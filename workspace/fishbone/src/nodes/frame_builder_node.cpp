#include <fishbone/nodes/frame_builder_node.hpp>

namespace FISHBONE
{

    FrameBuilderNode::FrameBuilderNode() : Node("frame_builder_node")
    {
        // 1. Parameters
        this->declare_parameter("use_stereo", true);
        this->declare_parameter("sequence", "00"); 
        
        this->is_stereo_mode_ = this->get_parameter("use_stereo").as_bool();
        std::string sequence = this->get_parameter("sequence").as_string();

        // 2. Load & Precompute Calibration (Run ONCE)
        this->loadCameraCalibrations(sequence);

        // 3. Initialize Publisher
        frame_publisher_ = this->create_publisher<fishbone::msg::Frame>("/fishbone/frame", 10);

        // 4. Initialize Subscribers
        sub_lidar_.subscribe(this, "/fishbone/lidar");

        if (this->is_stereo_mode_)
        {
            RCLCPP_INFO(this->get_logger(), "Initializing in STEREO Sync Mode");
            
            sub_stereo_.subscribe(this, "/fishbone/stereo_camera");

            sync_stereo_ = std::make_shared<message_filters::Synchronizer<StereoSyncPolicy>>(
                StereoSyncPolicy(10), sub_stereo_, sub_lidar_);
                
            sync_stereo_->registerCallback(
                std::bind(&FrameBuilderNode::stereoCallback, this, std::placeholders::_1, std::placeholders::_2));
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Initializing in MONO Sync Mode");
            
            sub_camera_.subscribe(this, "/fishbone/camera");

            sync_mono_ = std::make_shared<message_filters::Synchronizer<MonoSyncPolicy>>(
                MonoSyncPolicy(10), sub_camera_, sub_lidar_);

            sync_mono_->registerCallback(
                std::bind(&FrameBuilderNode::monoCallback, this, std::placeholders::_1, std::placeholders::_2));
        }
    }

    // --- Callback 1: Stereo + Lidar ---
    void FrameBuilderNode::stereoCallback(
        const fishbone::msg::StereoCameraData::ConstSharedPtr& stereo_msg,
        const fishbone::msg::LidarData::ConstSharedPtr& lidar_msg)
    {
        fishbone::msg::Frame frame;

        // Sync Info
        frame.frame_id = stereo_msg->left_camera.frame_id;
        frame.timestamp = stereo_msg->left_camera.timestamp;
        frame.is_stereo = true;

        // Copy Image Data
        frame.stereo_camera = *stereo_msg;
        frame.lidar_data = *lidar_msg;
        frame.has_lidar = true; 

        // OPTIMIZATION: Direct copy of pre-computed arrays (No Eigen math here!)
        frame.stereo_camera.left_camera.k = left_cam_static_.k;
        frame.stereo_camera.left_camera.p = left_cam_static_.p;
        
        frame.stereo_camera.right_camera.k = right_cam_static_.k;
        frame.stereo_camera.right_camera.p = right_cam_static_.p;
        
        frame_publisher_->publish(frame);
        
        RCLCPP_INFO_ONCE(this->get_logger(), "Publishing SYNCED Stereo+Lidar Frames...");
    }

    // --- Callback 2: Mono + Lidar ---
    void FrameBuilderNode::monoCallback(
        const fishbone::msg::CameraData::ConstSharedPtr& camera_msg,
        const fishbone::msg::LidarData::ConstSharedPtr& lidar_msg)
    {
        fishbone::msg::Frame frame;

        frame.frame_id = camera_msg->frame_id;
        frame.timestamp = camera_msg->timestamp;
        frame.is_stereo = false;

        frame.camera = *camera_msg;
        frame.lidar_data = *lidar_msg;
        frame.has_lidar = true;
        
        // OPTIMIZATION: Direct copy
        frame.camera.k = left_cam_static_.k;
        frame.camera.p = left_cam_static_.p;

        frame_publisher_->publish(frame);
        
        RCLCPP_INFO_ONCE(this->get_logger(), "Publishing SYNCED Mono+Lidar Frames...");
    }

    // --- Helper 1: Load Calib File (Run Once) ---
    void FrameBuilderNode::loadCameraCalibrations(std::string sequence) {
        std::string calib_path = "/workspace/data/dataset/sequences/" + sequence + "/calib.txt";
        std::ifstream file(calib_path);
        
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open calibration file: %s", calib_path.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Loading calibration from: %s", calib_path.c_str());

        std::vector<double> p2_raw, p3_raw;
        std::string line;
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix; 

            if (prefix == "P2:") {
                double val;
                while (ss >> val) p2_raw.push_back(val);
            } 
            else if (prefix == "P3:") {
                double val;
                while (ss >> val) p3_raw.push_back(val);
            }
        }
        
        // Validation & Precomputation
        if (p2_raw.size() == 12) {
            precomputeCameraInfo(left_cam_static_, p2_raw); // Store into static template
        } else {
             RCLCPP_ERROR(this->get_logger(), "Invalid P2 size: %zu", p2_raw.size());
        }

        if (p3_raw.size() == 12) {
            precomputeCameraInfo(right_cam_static_, p3_raw); // Store into static template
        } else {
             RCLCPP_ERROR(this->get_logger(), "Invalid P3 size: %zu", p3_raw.size());
        }
        
        file.close();
    }

    // --- Helper 2: Precompute (Run Once) ---
    void FrameBuilderNode::precomputeCameraInfo(fishbone::msg::CameraData& cam_msg, const std::vector<double>& p_raw) {
        if (p_raw.size() != 12) return;

        // 1. Map raw vector to Eigen Matrix (3x4)
        typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Matrix3x4d;
        Eigen::Map<const Matrix3x4d> P_eigen(p_raw.data());

        // 2. Extract K (Intrinsics) -> Top-left 3x3 block
        Eigen::Matrix3d K_eigen = P_eigen.block<3, 3>(0, 0);

        // 3. Store in the STATIC message object
        std::copy(p_raw.begin(), p_raw.end(), cam_msg.p.begin());
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K_msg_map(cam_msg.k.data());
        K_msg_map = K_eigen;
    }

} // namespace FISHBONE