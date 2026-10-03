#ifndef FISHBONE_VSLAM_RIGID_FIT_HPP_
#define FISHBONE_VSLAM_RIGID_FIT_HPP_

#include <Eigen/Dense>
#include <vector>

namespace FISHBONE {

// Closed-form rigid transform (R, t) minimizing sum |R*curr_i + t - prev_i|^2
// via SVD (Kabsch/Umeyama, no scale). Returns false if fewer than 3 points or
// the SVD is degenerate. Shared by Soft2MotionEstimator (frame-to-frame) and
// LoopClosureDetector (keyframe-to-keyframe) -- same underlying problem
// (align one set of 3D points onto another, known correspondence), same fit.
bool kabschFit(const std::vector<Eigen::Vector3d>& curr_pts,
                const std::vector<Eigen::Vector3d>& prev_pts,
                Eigen::Matrix3d& out_R, Eigen::Vector3d& out_t);

// Result of RANSAC + Kabsch registration between two point clouds with known
// (curr_pts[i] <-> prev_pts[i]) correspondence. R,t map curr onto prev:
// R*curr_i + t ~= prev_i -- i.e. this IS T_rel in this codebase's established
// convention (X_prev = R*X_curr + t).
struct RigidFitResult {
    bool ok = false;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    std::vector<int> inliers; // indices into curr_pts/prev_pts that supported the final fit
};

// RANSAC over 3-point minimal samples, Kabsch-fit each, count inliers within
// inlier_threshold_m (3D Euclidean residual), keep the model with the most
// inliers, then refit over all of that model's inliers for the final result.
RigidFitResult ransacKabsch(const std::vector<Eigen::Vector3d>& curr_pts,
                             const std::vector<Eigen::Vector3d>& prev_pts,
                             int iterations, double inlier_threshold_m,
                             int min_inliers, unsigned rng_seed = 12345u);

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_RIGID_FIT_HPP_
