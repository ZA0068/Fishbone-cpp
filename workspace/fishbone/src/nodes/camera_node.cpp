#include <fishbone/nodes/camera_node.hpp>

namespace FISHBONE
{

    CameraNode::CameraNode() : Node("camera_node"), camera_index_(0)
    {
        // Example: Initialize as Stereo. Change first bool to 'false' for mono.
        this->initializeCamera(false, true, true); 
    }

    void CameraNode::initializeCamera(bool stream, bool is_stereo, bool is_color)
    {
        this->is_stereo_mode_ = is_stereo;
        this->is_color_mode_ = is_color;

        if (stream)
        {
            RCLCPP_INFO(this->get_logger(), "Stream mode not implemented yet.");
        }
        else
        {
            this->loadImageFromFiles();

            // Only start the timer if images were actually found
            if (!left_image_paths_.empty()) {
                timer_ = this->create_wall_timer(
                    std::chrono::milliseconds(100), 
                    std::bind(&CameraNode::timer_callback_static, this));
            }
        }
    }

    // UPDATED: Now accepts 'is_stereo' flag to avoid failing if right folder is missing in mono mode
    std::pair<std::vector<std::string>, std::vector<std::string>> CameraNode::getImagePaths(const std::string &sequence)
    {
        std::vector<std::string> left_paths_out;
        std::vector<std::string> right_paths_out;
        
        auto [path_l, path_r] = this->imagePaths(sequence);

        // 1. Check Left Path (Always Required)
        if (!std::filesystem::exists(path_l))
        {
            RCLCPP_ERROR(this->get_logger(), "Left dataset path does not exist: %s", path_l.c_str());
            return {};
        }

        // 2. Check Right Path (Only Required if Stereo)
        if (this->is_stereo_mode_ && !std::filesystem::exists(path_r))
        {
            RCLCPP_ERROR(this->get_logger(), "Right dataset path does not exist (Stereo Mode): %s", path_r.c_str());
            return {};
        }

        // 3. Load Left Images
        for (const auto &entry : std::filesystem::directory_iterator(path_l))
        {
            if (entry.path().extension() == ".png")
            {
                left_paths_out.push_back(entry.path().string());
            }
        }
        std::sort(left_paths_out.begin(), left_paths_out.end());

        // 4. Load Right Images (Conditional)
        if (this->is_stereo_mode_)
        {
            for (const auto &entry : std::filesystem::directory_iterator(path_r))
            {
                if (entry.path().extension() == ".png")
                {
                    right_paths_out.push_back(entry.path().string());
                }
            }
            std::sort(right_paths_out.begin(), right_paths_out.end());

            // Validate sync
            if (left_paths_out.size() != right_paths_out.size()) {
                 RCLCPP_WARN(this->get_logger(), "Mismatch in image counts! Left: %zu, Right: %zu", 
                    left_paths_out.size(), right_paths_out.size());
            }
        }

        return {left_paths_out, right_paths_out};
    }


    void CameraNode::loadImageFromFiles()
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Dataset Mode. Stereo: %s", 
            is_stereo_mode_ ? "YES" : "NO");

        // ---------------------------------------------------------
        // 1. CONDITIONAL PUBLISHER INITIALIZATION
        // ---------------------------------------------------------
        if (is_stereo_mode_) {
            stereo_publisher_ = this->create_publisher<fishbone::msg::StereoCameraData>(
                "/fishbone/stereo_camera", 10);
        } else {
            mono_publisher_ = this->create_publisher<fishbone::msg::CameraData>(
                "/fishbone/camera", 10);
        }

        // 2. Load Paths
        this->declare_parameter("sequence", "00");
        std::string seq = this->get_parameter("sequence").as_string();
        
        auto [left_paths, right_paths] = this->getImagePaths(seq);
        this->left_image_paths_ = left_paths;
        this->right_image_paths_ = right_paths;

        if (left_image_paths_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No images found for sequence %s", seq.c_str());
        }
    }

    void CameraNode::timer_callback_static()
    {
        // Safety check
        if (left_image_paths_.empty()) return;

        // Stop once the final image has been published -- don't loop back to frame 0.
        // Actually shut down (not just cancel the timer) so this process exits and the
        // launch file's OnProcessExit handler can close the rest of the pipeline
        // (rviz, the feature window) instead of requiring Ctrl+C.
        if (camera_index_ >= left_image_paths_.size()) {
            RCLCPP_INFO(this->get_logger(), "Reached end of sequence (%zu images published), shutting down.",
                left_image_paths_.size());
            timer_->cancel();
            rclcpp::shutdown();
            return;
        }

        rclcpp::Time timestamp_ros = this->now();
        double timestamp_sec = timestamp_ros.seconds();

        // ---------------------------------------------------------
        // BRANCH 1: STEREO MODE
        // ---------------------------------------------------------
        if (this->is_stereo_mode_)
        {
            // A. Load Both Images
            cv::Mat img_l = cv::imread(left_image_paths_[camera_index_]);
            
            cv::Mat img_r;
            if (camera_index_ < right_image_paths_.size()) {
                img_r = cv::imread(right_image_paths_[camera_index_]);
            }

            if (img_l.empty() || img_r.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to read stereo pair %zu", camera_index_);
                camera_index_++;
                return;
            }

            // B. Build Stereo Message
            fishbone::msg::StereoCameraData msg;
            
            // --- Left Data ---
            msg.left_camera.frame_id = camera_index_;
            msg.left_camera.timestamp = timestamp_sec;
            msg.left_camera.width = img_l.cols;
            msg.left_camera.height = img_l.rows;
            
            // Convert Image
            std_msgs::msg::Header header_l;
            header_l.stamp = timestamp_ros;
            header_l.frame_id = "camera_left";
            cv_bridge::CvImage cv_bridge_l(header_l, "bgr8", img_l);
            msg.left_camera.image = *cv_bridge_l.toImageMsg();

            // Inject Calibration (Assuming p_left_ is populated)
            // std::copy(p_left_.begin(), p_left_.end(), msg.left.p.begin());

            // --- Right Data ---
            msg.right_camera.frame_id = camera_index_;
            msg.right_camera.timestamp = timestamp_sec;
            msg.right_camera.width = img_r.cols;
            msg.right_camera.height = img_r.rows;

            // Convert Image (Synced Timestamp)
            std_msgs::msg::Header header_r;
            header_r.stamp = timestamp_ros;
            header_r.frame_id = "camera_right";
            cv_bridge::CvImage cv_bridge_r(header_r, "bgr8", img_r);
            msg.right_camera.image = *cv_bridge_r.toImageMsg();

            // Inject Calibration
            // std::copy(p_right_.begin(), p_right_.end(), msg.right_camera.p.begin());

            // C. Publish Stereo
            stereo_publisher_->publish(msg);
        }
        // ---------------------------------------------------------
        // BRANCH 2: MONOCULAR MODE
        // ---------------------------------------------------------
        else 
        {
            // A. Load Left Image Only
            cv::Mat img = cv::imread(left_image_paths_[camera_index_]);

            if (img.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to read frame %zu", camera_index_);
                camera_index_++;
                return;
            }

            // B. Build Mono Message
            fishbone::msg::CameraData msg;
            
            msg.frame_id = camera_index_;
            msg.timestamp = timestamp_sec;
            msg.width = img.cols;
            msg.height = img.rows;

            std_msgs::msg::Header header;
            header.stamp = timestamp_ros;
            header.frame_id = "camera_mono";
            cv_bridge::CvImage cv_bridge(header, "bgr8", img);
            msg.image = *cv_bridge.toImageMsg();

            // Inject Calibration
            // std::copy(p_left_.begin(), p_left_.end(), msg.p.begin());

            // C. Publish Mono
            mono_publisher_->publish(msg);
        }

        camera_index_++;
    }

} // namespace FISHBONE