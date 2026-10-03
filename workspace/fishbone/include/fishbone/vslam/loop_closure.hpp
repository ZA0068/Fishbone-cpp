#ifndef FISHBONE_VSLAM_LOOP_CLOSURE_HPP_
#define FISHBONE_VSLAM_LOOP_CLOSURE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <functional>

namespace FISHBONE {

struct LoopClosureConfig {
    // AKAZE, not ORB: AKAZE's nonlinear (edge-preserving) scale space gives it
    // materially better repeatability under viewpoint/affine change than ORB's
    // Gaussian-pyramid + BRIEF, which matters specifically for KITTI's
    // reverse-direction revisits (approached from a different heading than
    // the original visit). Still a binary descriptor (MLDB), so the existing
    // NORM_HAMMING BFMatcher is unchanged. No new dependency -- built into
    // OpenCV's features2d, same module ORB came from.
    double akaze_threshold = 0.001;       // AKAZE detector response threshold -- lower = more keypoints
    double candidate_radius_m = 15.0;     // max current-pose distance to consider a past keyframe a candidate
    int min_keyframe_gap = 20;            // ignore keyframes fewer than this many indices back (too recent to be a "loop")
    int min_keyframes_between_loops = 5;     // rate-limit: skip searching again this soon (in keyframes) after the last accepted loop
    float match_ratio_test = 0.75f;       // Lowe's ratio test threshold for knnMatch
    int ransac_iterations = 500;
    double ransac_inlier_threshold_m = 0.5;
    int min_inliers = 15;                 // geometric verification acceptance gate
    double min_disparity_px = 2.0;        // same noise-floor reasoning as Soft2MotionConfig
    double max_depth_m = 50.0;
};

// A keyframe: a sparse AKAZE place-recognition signature plus the corresponding
// stereo-triangulated 3D point for each surviving keypoint, all in the
// keyframe's OWN camera frame (not world frame -- world placement is the pose
// graph's job, not this class's).
struct Keyframe {
    int index = -1;
    uint64_t frame_id = 0;
    Eigen::Matrix4d raw_pose = Eigen::Matrix4d::Identity(); // world_pose_ at creation time, never mutated
    std::vector<cv::KeyPoint> keypoints;    // only entries with a valid triangulated depth
    cv::Mat descriptors;                    // parallel to keypoints (AKAZE MLDB, one row per keypoint)
    std::vector<Eigen::Vector3d> points_local; // parallel to keypoints, in THIS keyframe's camera frame
};

// Detects loop closures: given a newly added keyframe, searches earlier
// keyframes for one close enough in (current best estimate of) world position
// to plausibly be the same place, verifies the match with AKAZE descriptor
// matching + RANSAC/Kabsch geometric verification (see rigid_fit.hpp -- same
// fit used for frame-to-frame odometry), and returns a pose-graph edge if
// verification succeeds.
class LoopClosureDetector {
public:
    explicit LoopClosureDetector(const LoopClosureConfig& config = LoopClosureConfig());

    // Extracts AKAZE keypoints on img_left, triangulates each via stereo (LK-matched
    // into img_right), keeps only ones with valid depth, and stores the result as
    // a new keyframe. `pose` is the current raw (odometry-only) world pose at this
    // instant -- the pose-graph node this keyframe will back is initialized from
    // it. Returns the new keyframe's index.
    int addKeyframe(const cv::Mat& img_left, const cv::Mat& img_right,
                     const Eigen::Matrix4d& pose, uint64_t frame_id,
                     double fx, double fy, double cx, double cy, double baseline);

    struct LoopEdge {
        int from = -1;    // older keyframe index
        int to = -1;      // newer keyframe index (the one just added)
        Eigen::Matrix4d T_rel = Eigen::Matrix4d::Identity(); // T_{from<-to}, same convention as Soft2MotionEstimator
        int inliers = 0;
    };

    // Searches for a loop closure involving keyframe `to` (normally the one just
    // returned by addKeyframe). `current_pose_lookup(idx)` must return keyframe
    // idx's best current pose estimate (graph-corrected if available, else its
    // raw_pose) -- proximity search uses this rather than raw_pose alone since
    // raw_pose drifts unboundedly over a long sequence. Returns the accepted edge,
    // or nullopt if no candidate was close enough or none passed verification.
    std::optional<LoopEdge> tryFindLoopClosure(
        int to, const std::function<Eigen::Matrix4d(int)>& current_pose_lookup);

    int size() const { return static_cast<int>(keyframes_.size()); }
    const Keyframe& keyframe(int i) const { return keyframes_.at(i); }

    // Diagnostics for tryFindLoopClosure's bail-out paths -- which of them is
    // firing tells apart "no candidate was ever close enough" from "a candidate
    // was found but AKAZE/RANSAC verification rejected it".
    struct Stats {
        int64_t attempts = 0;
        int64_t skipped_too_early = 0;        // to < min_keyframe_gap, or rate-limited
        int64_t skipped_no_own_descriptors = 0; // this keyframe itself had too few valid AKAZE points
        int64_t no_candidate_in_range = 0;     // no past keyframe within candidate_radius_m
        int64_t candidate_had_no_descriptors = 0;
        int64_t verification_failed_too_few_matches = 0; // ratio-test survivors below min_inliers
        int64_t verification_failed_ransac = 0;          // RANSAC/Kabsch itself didn't reach min_inliers
        int64_t accepted = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    LoopClosureConfig config_;
    std::vector<Keyframe> keyframes_;
    cv::Ptr<cv::AKAZE> akaze_;
    int last_loop_to_ = -1000000; // keyframe index of the most recently accepted loop's `to`, for rate-limiting
    Stats stats_;
};

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_LOOP_CLOSURE_HPP_
