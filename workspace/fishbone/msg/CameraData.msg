uint64 frame_id
float64 timestamp
sensor_msgs/Image image  # Maps to cv::Mat (use cv_bridge in code)
int32 width
int32 height

# Camera calibration matrices (Row-Major flattened)
float64[9] k    # Eigen::Matrix3d
float64[12] p   # Eigen::Matrix<double, 3, 4>

# Distortion coefficients
float64[] dist_coeffs