#ifndef FISHBONE_NODES_FRAME_BUILDER_NODE_HPP_
#define FISHBONE_NODES_FRAME_BUILDER_NODE_HPP_

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::copy

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <fishbone/msg/frame.hpp>
#include <fishbone/msg/camera_data.hpp>
#include <fishbone/msg/stereo_camera_data.hpp>
#include <fishbone/msg/lidar_data.hpp>

namespace FISHBONE
{
    class FrameBuilderNode : public rclcpp::Node
    {
    public:
        FrameBuilderNode();

    private:
        rclcpp::Publisher<fishbone::msg::Frame>::SharedPtr frame_publisher_;

        // --- Sync Policies ---
        typedef message_filters::sync_policies::ApproximateTime<
            fishbone::msg::CameraData, 
            fishbone::msg::LidarData
        > MonoSyncPolicy;

        typedef message_filters::sync_policies::ApproximateTime<
            fishbone::msg::StereoCameraData, 
            fishbone::msg::LidarData
        > StereoSyncPolicy;

        // Subscribers & Synchronizers
        message_filters::Subscriber<fishbone::msg::CameraData> sub_camera_;
        message_filters::Subscriber<fishbone::msg::StereoCameraData> sub_stereo_;
        message_filters::Subscriber<fishbone::msg::LidarData> sub_lidar_;

        std::shared_ptr<message_filters::Synchronizer<MonoSyncPolicy>> sync_mono_;
        std::shared_ptr<message_filters::Synchronizer<StereoSyncPolicy>> sync_stereo_;

        bool is_stereo_mode_;

        // --- OPTIMIZATION: Static Calibration Templates ---
        // We calculate these ONCE at startup, then just copy them.
        fishbone::msg::CameraData left_cam_static_;
        fishbone::msg::CameraData right_cam_static_;

        // --- Helper Methods ---
        void loadCameraCalibrations(std::string sequence);
        
        // Helper to process raw data into the static template (Run once)
        void precomputeCameraInfo(fishbone::msg::CameraData& cam_msg, const std::vector<double>& p_raw);

        // --- Callbacks ---
        void monoCallback(
            const fishbone::msg::CameraData::ConstSharedPtr& camera_msg,
            const fishbone::msg::LidarData::ConstSharedPtr& lidar_msg);

        void stereoCallback(
            const fishbone::msg::StereoCameraData::ConstSharedPtr& stereo_msg,
            const fishbone::msg::LidarData::ConstSharedPtr& lidar_msg);
    };
}

#endif // FISHBONE_NODES_FRAME_BUILDER_NODE_HPP_