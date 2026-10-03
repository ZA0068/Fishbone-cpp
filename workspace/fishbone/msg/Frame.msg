uint64 frame_id
float64 timestamp

# Camera data
CameraData camera
StereoCameraData stereo_camera

# Configuration flags
bool is_stereo  # true if stereo, false if monocular
bool is_color   # true if RGB, false if Grayscale

# GNSS/GPS data
GNSSData gnss_data
bool has_gnss

# IMU data
IMUData[] imu_measurements
bool has_imu

# LiDAR data
LidarData lidar_data
bool has_lidar


# Frame quality metrics
float32 feature_density
int32 num_tracked_features
bool is_keyframe