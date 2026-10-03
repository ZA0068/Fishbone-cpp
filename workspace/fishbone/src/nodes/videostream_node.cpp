#include <fishbone/nodes/videostream_node.hpp>

namespace FISHBONE {

VideoStreamNode::VideoStreamNode() : Node("video_stream_node")
{
    // Subscribe to the topic published by camera_node
    // Note: Use "fishbone/msg/frame" type
    subscription_ = this->create_subscription<fishbone::msg::Frame>(
        "/fishbone/frame", 10,
        std::bind(&VideoStreamNode::image_callback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "Video stream node has started.");
}

void VideoStreamNode::image_callback(const fishbone::msg::Frame::SharedPtr msg)
{
    try {
        // 1. Convert ROS Image message to OpenCV Mat
        // msg->left_camera is your custom 'CameraData' msg
        // msg->left_camera.image is the sensor_msgs/Image
        cv_bridge::CvImagePtr cv_ptr_left;
        cv_bridge::CvImagePtr cv_ptr_right;
        
        // We copy the image to BGR format
        cv_ptr_left = cv_bridge::toCvCopy(msg->stereo_camera.left_camera.image, sensor_msgs::image_encodings::BGR8);
        cv_ptr_right = cv_bridge::toCvCopy(msg->stereo_camera.right_camera.image, sensor_msgs::image_encodings::BGR8);

        // 2. Access the cv::Mat
        cv::Mat image_left = cv_ptr_left->image;
        cv::Mat image_right = cv_ptr_right->image;

        if (image_left.empty() || image_right.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty image frame!");
            return;
        }

        // 3. Display
        cv::namedWindow("Left camera Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Left camera Image", image_left);
        cv::waitKey(1); // Required to draw the window

        cv::namedWindow("Right camera Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Right camera Image", image_right);
        cv::waitKey(1); // Required to draw the window

    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}

}