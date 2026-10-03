#ifndef FISHBONE_VSLAM_VISUAL_ODOMETRY_HPP_
#define FISHBONE_VSLAM_VISUAL_ODOMETRY_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace FISHBONE {

class VisualOdometry {
public:
    // Ptr typedef for easy shared pointer usage
    typedef std::shared_ptr<VisualOdometry> Ptr;

    VisualOdometry();
    ~VisualOdometry();

    /**
     * @brief Main function to process a new stereo frame
     * @param img_left  Left camera image (grayscale)
     * @param img_right Right camera image (grayscale)
     * @return cv::Mat  Image with drawn matches for debugging
     */
    cv::Mat processFrame(const cv::Mat& img_left, const cv::Mat& img_right);

private:
    // Feature Detector (ORB)
    cv::Ptr<cv::FeatureDetector> detector_;
    
    // Feature Matcher (BFMatcher - Brute Force with Hamming distance)
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    // Data from the PREVIOUS frame (for temporal tracking)
    std::vector<cv::KeyPoint> keypoints_last_left_;
    cv::Mat descriptors_last_left_;
    cv::Mat last_image_left_;

    // Function to match features between two sets of descriptors
    std::vector<cv::DMatch> matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2);
};

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_VISUAL_ODOMETRY_HPP_