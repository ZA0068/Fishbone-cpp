#ifndef FISHBONE_VSLAM_SOFT2_MOTION_HPP_
#define FISHBONE_VSLAM_SOFT2_MOTION_HPP_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <fishbone/vslam/fishbone.hpp>
#include <fishbone/vslam/rigid_fit.hpp>
#include <fishbone/msg/lidar_point.hpp>

namespace FISHBONE {

// Tunables for the motion estimator.
//
// This used to be 2D-3D PnP (solvePnPRansac). Replaced with RANSAC + Kabsch/SVD
// 3D-3D rigid-transform fitting: since we're stereo, every tracked feature
// already has real triangulated depth at BOTH t and t-1 (pt/pt_r_curr and
// pt_l_prev/pt_r_prev) -- there's no need to go back through a 2D projection.
// PnP's failure mode was that it solves rotation and translation together in
// one 6-DOF Jacobian, so whenever translation was poorly conditioned (constant
// on KITTI: near-pure forward motion, SQPnP's point_coordinate_variance
// assertion was firing on ~60-80% of frames), the WHOLE solve failed and
// rotation got discarded along with it -- even on frames where the vehicle was
// genuinely turning and rotation was easily recoverable from the same points.
// Kabsch's closed-form SVD fit separates the two cleanly: R comes from the
// point clouds' cross-covariance alone, t = centroid_prev - R*centroid_curr
// falls out after. It stays well-conditioned as long as points aren't
// literally collinear in 3D -- far more forgiving than PnP's coupled Jacobian
// -- and rotation is recovered on every accepted frame, not gated behind a
// translation-specific degeneracy check.
struct Soft2MotionConfig {
    int ransac_iterations = 300;              // RANSAC iterations, each a 3-point Kabsch fit (cheap: closed-form SVD)
    double ransac_inlier_threshold_m = 0.3;   // 3D Euclidean residual gate for inlier counting, meters
    int min_points = 6;                       // minimum RANSAC inliers to accept a solve
    double max_plausible_translation_m = 5.0; // safety bound: reject solves beyond this per-frame translation

    // Numerical-conditioning tunables, added after diagnosing why ~80% of frames
    // were failing PnP: near-zero-disparity ("noise floor") and very-far points
    // were dominating/polluting the triangulated point cloud. Still apply here --
    // Kabsch is more forgiving of geometric conditioning, but garbage depth is
    // still garbage depth.
    double min_disparity_px = 2.0;     // reject a triangulated point below this stereo disparity (noise floor)
    double max_depth_m = 50.0;         // far points are dominated by disparity noise (Z = f*b/d blows up as d->0)
    double min_point_spread_m = 1.0;   // correspondence cloud's X or Y stddev must clear this before RANSAC at all
    int max_coast_frames = 10;         // cap on consecutive frames allowed to coast on the last accepted motion
};

// Diagnostic counters: which of estimate()'s bail-out paths (each of which
// coasts on the last accepted motion, see fallback()) is firing, and how often.
struct Soft2MotionStats {
    int64_t calls = 0;
    int64_t reject_too_few_features = 0;          // features.size() < 10 gate
    int64_t reject_too_few_correspondences = 0;   // not enough valid 3D-3D pairs after triangulating both frames
    int64_t reject_low_point_spread = 0;           // correspondence cloud too collinear/coplanar, skipped RANSAC entirely
    int64_t reject_too_few_inliers = 0;            // RANSAC never found a model with >= min_points inliers
    int64_t reject_translation_too_large = 0;      // sanity bound rejected an implausible solve
    int64_t reject_exception = 0;                  // caught an exception (defensive; Kabsch itself shouldn't throw)
    int64_t accepted = 0;

    // Every reject_* path above coasts on the last accepted motion instead of
    // freezing to Identity (see estimate()'s fallback()). These count that.
    int64_t coasted = 0;         // a reject_* path fired, but we were still within max_coast_frames
    int64_t coast_capped = 0;    // a reject_* path fired AND we'd already coasted max_coast_frames in a row

    // Extra detail for the translation_too_large rejections specifically, to tell
    // a moderate miscalibration apart from a genuine numerical blowup.
    double sum_rejected_translation_m = 0.0;
    double max_rejected_translation_m = 0.0;
    int64_t sum_rejected_inliers = 0;
    int64_t sum_rejected_correspondences = 0;

    // Same detail for accepted solves, for comparison.
    double sum_accepted_translation_m = 0.0;
    int64_t sum_accepted_inliers = 0;
    int64_t sum_accepted_correspondences = 0;
};

class Soft2MotionEstimator {
public:
    Soft2MotionEstimator(const Soft2MotionConfig& config = Soft2MotionConfig());

    /**
     * @brief Estimates motion via RANSAC + Kabsch 3D-3D rigid registration between
     * each tracked feature's stereo-triangulated position at t and t-1 (lidar depth
     * preferred at t when available; stereo triangulation otherwise/always at t-1).
     *
     * @param features      Visual features from Circular Matching
     * @param K             Camera Intrinsics
     * @param lidar_points  Raw Lidar points (in Lidar Frame)
     * @param T_cam_lidar   Extrinsics: Transform from Lidar Frame to Camera Frame (4x4)
     * @param baseline      Stereo baseline
     */
    Eigen::Matrix4d estimate(
        const std::vector<FeaturePoint>& features,
        const cv::Mat& K,
        const std::vector<fishbone::msg::LidarPoint>& lidar_points,
        const Eigen::Matrix4d& T_cam_lidar,
        double baseline);

    const Soft2MotionStats& stats() const { return stats_; }

private:
    Soft2MotionConfig config_;
    Soft2MotionStats stats_;

    // Constant-velocity coasting state: the last accepted relative motion, and how
    // many consecutive frames we've been repeating it for (see estimate()'s fallback()).
    Eigen::Matrix4d last_valid_T_rel_ = Eigen::Matrix4d::Identity();
    int consecutive_stale_ = 0;

    cv::Point3d triangulateStereo(const cv::Point2f& pt_l, const cv::Point2f& pt_r,
                                 double f, double cx, double cy, double b);

    // NEW: Helper to find Lidar depth for a specific 2D pixel
    double findLidarDepth(const cv::Point2f& pt, const cv::Mat& depth_map);

    // NEW: Projects Lidar cloud into a sparse depth image for fast lookup
    cv::Mat projectLidarToImage(
        const std::vector<fishbone::msg::LidarPoint>& points,
        const Eigen::Matrix4d& T_cam_lidar,
        const cv::Mat& K,
        int img_width, int img_height);
};

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_SOFT2_MOTION_HPP_
