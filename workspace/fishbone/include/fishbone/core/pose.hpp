// fishbone/core/pose.hpp
#ifndef FISHBONE_CORE_POSE_HPP_
#define FISHBONE_CORE_POSE_HPP_

#include <cstdint>
#include <Eigen/Dense>

namespace FISHBONE {

struct StateEstimate {
    uint64_t frame_id;  // Corresponds to Frame::frame_id
    double timestamp;
    
    // 1. World Pose (T_wb: Body to World)
    // Using 4x4 Matrix for SE(3)
    Eigen::Matrix4d pose; 
    
    // 2. Motion Derivatives (Body Frame)
    Eigen::Vector3d velocity_linear;  // v_b (m/s)
    Eigen::Vector3d velocity_angular; // w_b (rad/s)
    
    // 3. Graph/Optimization Metadata (Optional but recommended)
    bool is_keyframe;            // Was this added to the Factor Graph?
    double optimization_error;   // Residual error from GTSAM/Solver
    
    // 4. Covariance (Uncertainty)
    // 6x6 Matrix (Rot, Trans)
    Eigen::Matrix<double, 6, 6> covariance;

    StateEstimate() 
        : frame_id(0), timestamp(0.0),
          is_keyframe(false), optimization_error(0.0)
    {
        pose.setIdentity();
        velocity_linear.setZero();
        velocity_angular.setZero();
        covariance.setZero();
    }
};

} // namespace FISHBONE

#endif // FISHBONE_CORE_POSE_HPP_