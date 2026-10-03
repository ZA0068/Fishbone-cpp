#include <fishbone/vslam/soft2_motion.hpp>
#include <numeric>
#include <algorithm>
#include <iostream> // Added for logging errors

namespace FISHBONE {

Soft2MotionEstimator::Soft2MotionEstimator(const Soft2MotionConfig& config)
    : config_(config)
{
}

Eigen::Matrix4d Soft2MotionEstimator::estimate(
    const std::vector<FeaturePoint>& features,
    const cv::Mat& K,
    const std::vector<fishbone::msg::LidarPoint>& lidar_points,
    const Eigen::Matrix4d& T_cam_lidar,
    double baseline)
{
    stats_.calls++;

    // Constant-velocity coast: every reject_* path below calls this instead of
    // returning Identity() directly. For a continuously-moving platform, "repeat
    // the last known motion" is a far better prior than "the car stopped dead" --
    // Identity-as-failure-default is what turned a high rejection rate into a
    // trajectory that flatlines at the origin most of the time (confirmed by
    // plotting estimated-vs-ground-truth per-frame translation). Capped at
    // max_coast_frames so a genuinely stuck/degenerate run doesn't drift forever
    // on a stale extrapolation.
    auto fallback = [this]() -> Eigen::Matrix4d {
        if (consecutive_stale_ < config_.max_coast_frames) {
            consecutive_stale_++;
            stats_.coasted++;
            return last_valid_T_rel_;
        }
        stats_.coast_capped++;
        return Eigen::Matrix4d::Identity();
    };

    if (features.size() < 10) { stats_.reject_too_few_features++; return fallback(); }

    // Pre-processing: Project Lidar to Image Plane.
    // A velodyne scan is ~125k points; projecting all of them is real work (matrix
    // transform + projection per point, every single frame). T_cam_lidar == Identity
    // is the sentinel this codebase uses for "no real lidar-camera calibration
    // available" (see fishbone_node.cpp) -- in that case the projected points don't
    // land anywhere meaningful anyway, so skip the pass entirely rather than pay for
    // a depth map that findLidarDepth would mostly fail to get hits from regardless.
    cv::Mat lidar_depth_map;
    if (!T_cam_lidar.isApprox(Eigen::Matrix4d::Identity())) {
        lidar_depth_map = projectLidarToImage(lidar_points, T_cam_lidar, K, 1241, 376);
    } else {
        lidar_depth_map = cv::Mat::zeros(376, 1241, CV_32F);
    }

    // 3D-3D rigid registration instead of PnP: triangulate each feature's position
    // in BOTH the current camera's frame (lidar-preferred, matching this codebase's
    // established depth-source priority) and the previous camera's frame (stereo
    // triangulation -- no retained lidar scan for the prior frame), then fit the
    // rigid transform that aligns one point set onto the other. See the header's
    // Soft2MotionConfig doc comment for why this replaced PnP.
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double f = (fx + fy) / 2.0;

    std::vector<Eigen::Vector3d> curr_pts, prev_pts; // parallel arrays, in each camera's OWN frame

    for (const auto& ft : features) {
        double Z_curr = -1.0;

        // Try lidar first, fall back to stereo triangulation, for the CURRENT point.
        double lidar_z = findLidarDepth(ft.pt, lidar_depth_map);
        if (lidar_z > 0.1) {
            Z_curr = lidar_z;
            if (Z_curr <= 0.1 || Z_curr > config_.max_depth_m) continue;
            double X = (ft.pt.x - cx) * Z_curr / f;
            double Y = (ft.pt.y - cy) * Z_curr / f;
            cv::Point3d P_prev = triangulateStereo(ft.pt_l_prev, ft.pt_r_prev, f, cx, cy, baseline);
            if (P_prev.z <= 0.1 || P_prev.z > config_.max_depth_m) continue;
            curr_pts.emplace_back(X, Y, Z_curr);
            prev_pts.emplace_back(P_prev.x, P_prev.y, P_prev.z);
            continue;
        }

        cv::Point3d P_curr = triangulateStereo(ft.pt, ft.pt_r_curr, f, cx, cy, baseline);
        if (P_curr.z <= 0.1 || P_curr.z > config_.max_depth_m) continue;
        cv::Point3d P_prev = triangulateStereo(ft.pt_l_prev, ft.pt_r_prev, f, cx, cy, baseline);
        if (P_prev.z <= 0.1 || P_prev.z > config_.max_depth_m) continue;

        curr_pts.emplace_back(P_curr.x, P_curr.y, P_curr.z);
        prev_pts.emplace_back(P_prev.x, P_prev.y, P_prev.z);
    }

    if (static_cast<int>(curr_pts.size()) < config_.min_points) {
        stats_.reject_too_few_correspondences++;
        return fallback();
    }

    // Proactive conditioning check, done BEFORE spending RANSAC iterations. Forward-
    // driving KITTI scenes are the classic degenerate case: most tracked points sit
    // near the direction of travel / epipole. Kabsch tolerates this far better than
    // PnP (rotation is recoverable independent of translation conditioning), but a
    // literally collinear point set (no lateral X or Y spread at all) still can't
    // constrain a unique rigid transform.
    {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& p : curr_pts) centroid += p;
        centroid /= static_cast<double>(curr_pts.size());
        double var_x = 0.0, var_y = 0.0;
        for (const auto& p : curr_pts) {
            double dx = p.x() - centroid.x(), dy = p.y() - centroid.y();
            var_x += dx * dx;
            var_y += dy * dy;
        }
        double std_x = std::sqrt(var_x / curr_pts.size());
        double std_y = std::sqrt(var_y / curr_pts.size());
        if (std_x < config_.min_point_spread_m && std_y < config_.min_point_spread_m) {
            stats_.reject_low_point_spread++;
            return fallback();
        }
    }

    try {
        const size_t n = curr_pts.size();
        RigidFitResult fit = ransacKabsch(curr_pts, prev_pts, config_.ransac_iterations,
                                           config_.ransac_inlier_threshold_m, config_.min_points,
                                           12345u + static_cast<unsigned>(n));

        // ransacKabsch reports failure the same way whether RANSAC never found a
        // usable 3-point sample or found one but its best inlier count fell short
        // of min_points -- both mean "not enough agreement in this point set",
        // so both count as reject_too_few_inliers.
        if (!fit.ok) { stats_.reject_too_few_inliers++; return fallback(); }

        const Eigen::Matrix3d& R = fit.R;
        const Eigen::Vector3d& t = fit.t;
        int best_inlier_count = static_cast<int>(fit.inliers.size());
        double t_norm = t.norm();

        // SAFETY: sanity-bound the result. At 10 Hz even a fast car moves at most
        // ~1-2 m/frame; a solved translation far beyond that means something is
        // still wrong (e.g. a majority-inlier degenerate sample outvoting the truth).
        if (t_norm > config_.max_plausible_translation_m) {
            stats_.reject_translation_too_large++;
            stats_.sum_rejected_translation_m += t_norm;
            stats_.max_rejected_translation_m = std::max(stats_.max_rejected_translation_m, t_norm);
            stats_.sum_rejected_inliers += best_inlier_count;
            stats_.sum_rejected_correspondences += static_cast<int64_t>(n);
            return fallback();
        }

        // Kabsch was fit to align curr_pts onto prev_pts, i.e. R*P_curr + t ~= P_prev
        // -- exactly T_rel's documented convention (X_prev = R*X_curr + t), no
        // conversion needed.
        Eigen::Matrix4d T_rel = Eigen::Matrix4d::Identity();
        T_rel.block<3, 3>(0, 0) = R;
        T_rel.block<3, 1>(0, 3) = t;

        stats_.accepted++;
        stats_.sum_accepted_translation_m += t_norm;
        stats_.sum_accepted_inliers += best_inlier_count;
        stats_.sum_accepted_correspondences += static_cast<int64_t>(n);
        last_valid_T_rel_ = T_rel;
        consecutive_stale_ = 0;
        return T_rel;

    } catch (const std::exception& e) {
        std::cerr << "[Soft2Motion] Error: " << e.what() << std::endl;
        stats_.reject_exception++;
        return fallback();
    }
}

cv::Mat Soft2MotionEstimator::projectLidarToImage(
    const std::vector<fishbone::msg::LidarPoint>& points,
    const Eigen::Matrix4d& T_cam_lidar,
    const cv::Mat& K,
    int img_width, int img_height)
{
    cv::Mat depth_map = cv::Mat::zeros(img_height, img_width, CV_32F);
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    for (const auto& pt : points) {
        Eigen::Vector4d P_lidar(pt.x, pt.y, pt.z, 1.0);
        Eigen::Vector4d P_cam = T_cam_lidar * P_lidar;

        if (P_cam.z() <= 0.1) continue;

        int u = std::round(fx * P_cam.x() / P_cam.z() + cx);
        int v = std::round(fy * P_cam.y() / P_cam.z() + cy);

        if (u >= 0 && u < img_width && v >= 0 && v < img_height) {
            float current = depth_map.at<float>(v, u);
            if (current == 0.0f || P_cam.z() < current) {
                depth_map.at<float>(v, u) = (float)P_cam.z();
            }
        }
    }
    return depth_map;
}

double Soft2MotionEstimator::findLidarDepth(const cv::Point2f& pt, const cv::Mat& depth_map) {
    int u = std::round(pt.x);
    int v = std::round(pt.y);
    int radius = 2;
    double min_dist = 9999.0;
    bool found = false;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int ny = v + dy;
            int nx = u + dx;

            if (nx >= 0 && nx < depth_map.cols && ny >= 0 && ny < depth_map.rows) {
                float d = depth_map.at<float>(ny, nx);
                if (d > 0.1f) {
                    if (d < min_dist) min_dist = d;
                    found = true;
                }
            }
        }
    }
    return found ? min_dist : -1.0;
}

cv::Point3d Soft2MotionEstimator::triangulateStereo(
    const cv::Point2f& pt_l, const cv::Point2f& pt_r,
    double f, double cx, double cy, double b)
{
    double d = pt_l.x - pt_r.x;
    // Was `abs(d) < 0.1` -- disparity noise alone is typically +-0.1-0.5px, so
    // points right at that old gate were essentially noise-dominated depth
    // (Z = f*b/d blows up as d->0), and were exactly the kind of numerically
    // toxic point polluting the point cloud PnP was then rejecting for low
    // spatial variance. min_disparity_px (default 2.0) is a real noise floor.
    if (d < config_.min_disparity_px) return cv::Point3d(0, 0, -1.0);
    double Z = (f * b) / d;
    return cv::Point3d((pt_l.x - cx) * Z / f, (pt_l.y - cy) * Z / f, Z);
}

} // namespace FISHBONE
