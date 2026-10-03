#include <fishbone/vslam/loop_closure.hpp>
#include <fishbone/vslam/rigid_fit.hpp>
#include <opencv2/video/tracking.hpp>

namespace FISHBONE {

LoopClosureDetector::LoopClosureDetector(const LoopClosureConfig& config)
    : config_(config)
{
    // MLDB descriptor (binary, like ORB's BRIEF) -- keeps the existing
    // NORM_HAMMING BFMatcher valid unchanged. See LoopClosureConfig's doc
    // comment for why AKAZE over ORB here.
    akaze_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3,
        static_cast<float>(config_.akaze_threshold));
}

int LoopClosureDetector::addKeyframe(const cv::Mat& img_left, const cv::Mat& img_right,
                                      const Eigen::Matrix4d& pose, uint64_t frame_id,
                                      double fx, double fy, double cx, double cy, double baseline)
{
    std::vector<cv::KeyPoint> raw_kps;
    cv::Mat raw_descs;
    akaze_->detectAndCompute(img_left, cv::noArray(), raw_kps, raw_descs);

    Keyframe kf;
    kf.index = static_cast<int>(keyframes_.size());
    kf.frame_id = frame_id;
    kf.raw_pose = pose;

    if (!raw_kps.empty()) {
        // Single-shot LK from the left keypoints into the right image -- this is a
        // one-off stereo correspondence search (not temporal tracking), the same
        // idea as Fishbone::circularMatch's step 3 but only ever run once per
        // keyframe, so it's cheap to duplicate here rather than reach into that
        // module's internals.
        std::vector<cv::Point2f> pts_l;
        pts_l.reserve(raw_kps.size());
        for (const auto& kp : raw_kps) pts_l.push_back(kp.pt);

        std::vector<cv::Point2f> pts_r;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(img_left, img_right, pts_l, pts_r, status, err,
                                  cv::Size(21, 21), 3,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

        const double f = (fx + fy) / 2.0;
        cv::Mat filtered_descs;
        filtered_descs.reserve(raw_kps.size());

        for (size_t i = 0; i < raw_kps.size(); ++i) {
            if (!status[i]) continue;
            double disparity = pts_l[i].x - pts_r[i].x;
            if (disparity < config_.min_disparity_px) continue;
            double Z = (f * baseline) / disparity;
            if (Z <= 0.1 || Z > config_.max_depth_m) continue;

            double X = (pts_l[i].x - cx) * Z / f;
            double Y = (pts_l[i].y - cy) * Z / f;

            kf.keypoints.push_back(raw_kps[i]);
            kf.points_local.emplace_back(X, Y, Z);
            filtered_descs.push_back(raw_descs.row(static_cast<int>(i)));
        }
        kf.descriptors = filtered_descs;
    }

    keyframes_.push_back(std::move(kf));
    return static_cast<int>(keyframes_.size()) - 1;
}

std::optional<LoopClosureDetector::LoopEdge> LoopClosureDetector::tryFindLoopClosure(
    int to, const std::function<Eigen::Matrix4d(int)>& current_pose_lookup)
{
    if (to < 0 || to >= static_cast<int>(keyframes_.size())) return std::nullopt;
    stats_.attempts++;
    if (to < config_.min_keyframe_gap) { stats_.skipped_too_early++; return std::nullopt; } // not enough history yet
    if (to - last_loop_to_ < config_.min_keyframes_between_loops) { stats_.skipped_too_early++; return std::nullopt; } // rate limit

    const Keyframe& kf_to = keyframes_[to];
    if (kf_to.descriptors.empty() || kf_to.descriptors.rows < config_.min_inliers) {
        stats_.skipped_no_own_descriptors++;
        return std::nullopt;
    }

    Eigen::Vector3d pos_to = current_pose_lookup(to).block<3, 1>(0, 3);

    // Nearest-by-position candidate among keyframes old enough to plausibly be a
    // revisit, not just the immediately preceding stretch of the current path.
    int best_candidate = -1;
    double best_dist = config_.candidate_radius_m;
    for (int i = 0; i <= to - config_.min_keyframe_gap; ++i) {
        Eigen::Vector3d pos_i = current_pose_lookup(i).block<3, 1>(0, 3);
        double dist = (pos_i - pos_to).norm();
        if (dist < best_dist) {
            best_dist = dist;
            best_candidate = i;
        }
    }
    if (best_candidate < 0) { stats_.no_candidate_in_range++; return std::nullopt; }

    const Keyframe& kf_from = keyframes_[best_candidate];
    if (kf_from.descriptors.empty() || kf_from.descriptors.rows < config_.min_inliers) {
        stats_.candidate_had_no_descriptors++;
        return std::nullopt;
    }

    // Descriptor matching: query = kf_to (the newer keyframe), train = kf_from.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(kf_to.descriptors, kf_from.descriptors, knn_matches, 2);

    std::vector<Eigen::Vector3d> curr_pts, prev_pts; // curr=kf_to's points, prev=kf_from's points
    for (const auto& m : knn_matches) {
        if (m.size() < 2) continue;
        if (m[0].distance >= config_.match_ratio_test * m[1].distance) continue; // Lowe's ratio test
        curr_pts.push_back(kf_to.points_local[static_cast<size_t>(m[0].queryIdx)]);
        prev_pts.push_back(kf_from.points_local[static_cast<size_t>(m[0].trainIdx)]);
    }
    if (static_cast<int>(curr_pts.size()) < config_.min_inliers) {
        stats_.verification_failed_too_few_matches++;
        return std::nullopt;
    }

    RigidFitResult fit = ransacKabsch(curr_pts, prev_pts, config_.ransac_iterations,
                                       config_.ransac_inlier_threshold_m, config_.min_inliers);
    if (!fit.ok) { stats_.verification_failed_ransac++; return std::nullopt; }

    stats_.accepted++;
    LoopEdge edge;
    edge.from = best_candidate;
    edge.to = to;
    edge.T_rel = Eigen::Matrix4d::Identity();
    edge.T_rel.block<3, 3>(0, 0) = fit.R;
    edge.T_rel.block<3, 1>(0, 3) = fit.t;
    edge.inliers = static_cast<int>(fit.inliers.size());

    last_loop_to_ = to;
    return edge;
}

} // namespace FISHBONE
