uint64 frame_id
float64 timestamp

# 1. World Pose (T_wb: Body to World)
# 4x4 Matrix for SE(3), flattened in Row-Major order
float64[16] pose

# 2. Motion Derivatives (Body Frame)
# [x, y, z] in m/s
float64[3] velocity_linear

# [x, y, z] in rad/s
float64[3] velocity_angular

# [x, y, z] in m/s^2
float64[3] acceleration_linear

# [x, y, z] in rad/s^2
float64[3] acceleration_angular

# 3. Graph/Optimization Metadata
bool is_keyframe
float64 optimization_error

# 4. Covariance (Uncertainty)
# 6x6 Matrix (Rotation, Translation), flattened in Row-Major order
float64[36] covariance