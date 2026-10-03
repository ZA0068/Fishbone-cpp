#ifndef FISHBONE_NODES_FISHBONE_NODE_HPP_
#define FISHBONE_NODES_FISHBONE_NODE_HPP_

#include <fstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/path.hpp>

#include <cv_bridge/cv_bridge.hpp> 
#include <image_transport/image_transport.hpp>

#include <fishbone/msg/frame.hpp>
#include <fishbone/msg/pose_data.hpp>
#include <fishbone/vslam/fishbone.hpp>
#include <fishbone/vslam/soft2_motion.hpp>
#include <fishbone/vslam/loop_closure.hpp>
#include <fishbone/vslam/pose_graph.hpp>

namespace FISHBONE {

class FishboneNode : public rclcpp::Node {
public:
    FishboneNode();
    ~FishboneNode();

private:
    void frameCallback(const fishbone::msg::Frame::SharedPtr msg);
    void publishFeatureVisualization(const cv::Mat& left_img, const rclcpp::Time& stamp);

    // step 10 helpers
    std::vector<Eigen::Matrix4d> loadGroundTruthPoses(const std::string& path) const;
    void publishPaths(const rclcpp::Time& stamp);
    void updateAndLogError(uint64_t frame_id, const Eigen::Matrix4d& T_rel);

    // --- Loop closure / global map (pose graph) ---
    //
    // world_pose_ (below) is a pure LOCAL/raw odometry chain: it accumulates
    // T_rel every frame and is NEVER touched by graph optimization -- it stays
    // the live, low-latency estimate used for /fishbone/pose, velocity/
    // acceleration, and the incremental path publish. Global correction is
    // layered on top, additively, via FrameRecord + PoseGraph:
    //
    //   every frame gets a FrameRecord{keyframe_idx, T_local}, where T_local is
    //   its pose relative to the most recent keyframe (both computed from the
    //   raw/uncorrected world_pose_ chain, so T_local never needs to change
    //   once written); the frame's globally-corrected pose is then always
    //   `pose_graph_.pose(keyframe_idx) * T_local` -- correctedPose() below.
    //
    // A loop closure only ever changes keyframe nodes' pose_graph_ estimates;
    // since every frame's corrected pose is *computed* from those (not stored
    // independently), a single optimize() call retroactively corrects every
    // already-passed frame's output for free, without re-running VO.
    struct FrameRecord {
        uint64_t frame_id;
        int keyframe_idx;       // index into loop_closure_'s keyframe list (and, 1:1, pose_graph_'s nodes)
        Eigen::Matrix4d T_local; // this frame's pose relative to that keyframe's raw_pose
    };

    Eigen::Matrix4d correctedPose(const FrameRecord& rec) const;
    void recordFramePose(uint64_t frame_id);
    // Adds a new keyframe when the keyframe-trigger policy fires, wires it into
    // the pose graph (odometry edge from the previous keyframe), searches for a
    // loop closure, and -- if one is found -- optimizes the graph and rewrites
    // the corrected output (pose file + RViz path). Takes the RAW (unfoveated)
    // image pair -- ORB place-recognition descriptors need consistent sharpness
    // regardless of where a point falls in frame, whereas the tracking pipeline's
    // foveated images are deliberately blurred away from center.
    void maybeAddKeyframeAndCloseLoop(const cv::Mat& raw_left, const cv::Mat& raw_right,
        uint64_t frame_id, double fx, double fy, double cx, double cy, double baseline);
    // Full rewrite of /workspace/generated/<sequence>.txt from frame_records_ +
    // pose_graph_'s current (possibly optimized) keyframe poses. Cheap enough
    // (thousands of lines) to just redo from scratch rather than patch in place.
    void writePoseFile();
    // Rebuilds path_estimated_ from scratch using corrected poses and republishes
    // it -- called right after a loop-closure optimize() so RViz visibly reflects
    // the correction, instead of only affecting the on-disk output.
    void republishCorrectedPath();

    // Angular velocity/acceleration from T_rel's rotation and its finite difference.
    void computeAngularMotion(const Eigen::Matrix4d& T_rel, double dt,
        Eigen::Vector3d& out_vel_ang, Eigen::Vector3d& out_acc_ang);

    // Linear velocity/acceleration derived directly from tracked-feature triangulation
    // rather than from the (noisy) essential-matrix pose estimate:
    //   velocity     <- ipsi-temporal   (L<->L', R<->R'): P(t) vs P(t-1)
    //   acceleration <- cross-temporal  (L'<->R'', R'<->L''): the P(t-1)-vs-P(t-2)
    //                    velocity compared against the ipsi one above.
    // Only features with has_cross_history (i.e. L, L', L'', R, R', R'' all matched
    // this round) participate; per-feature estimates are combined via median.
    void computeLinearMotionFromFeatures(double fx, double cx, double cy, double baseline,
        double dt, Eigen::Vector3d& out_vel_lin, Eigen::Vector3d& out_acc_lin) const;
    
    std::unique_ptr<Fishbone> fishbone_;
    std::unique_ptr<Soft2MotionEstimator> estimator_;
    std::unique_ptr<LoopClosureDetector> loop_closure_;
    std::unique_ptr<PoseGraph> pose_graph_;

    // Whether the live cv::imshow debug window could be opened (false if no X
    // display/auth is available -- pose estimation shouldn't depend on it).
    bool gui_available_ = false;

    rclcpp::Subscription<fishbone::msg::Frame>::SharedPtr frame_sub_;
    rclcpp::Publisher<fishbone::msg::PoseData>::SharedPtr pose_pub_;
    image_transport::Publisher feature_image_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_estimated_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_ground_truth_pub_;

    // --- Tracking state carried between callbacks: rolling 3-frame window ---
    // NOTE: these hold FOVEATED images (see Fishbone::foveate) -- detection, matching,
    // and tracking all operate on the foveated L/R pair, not the raw camera frames.
    bool has_prev_frame_;
    uint64_t last_seen_frame_id_ = 0; // diagnostic: detects dropped/skipped frame_ids (see frameCallback)
    cv::Mat prev_left_, prev_right_;           // t-1
    cv::Mat prev_prev_left_, prev_prev_right_; // t-2 (empty until the 3rd real frame)
    std::vector<FeaturePoint> features_;

    // Count from the last detect() call (bootstrap or replenish); the replenish
    // threshold is 50% of this, not a fixed constant.
    size_t baseline_feature_count_ = 0;

    // Accumulated camera pose in world frame (T_wc)
    Eigen::Matrix4d world_pose_;

    // Angular motion history, for acceleration's finite difference
    bool has_prev_angular_velocity_ = false;
    Eigen::Vector3d prev_velocity_angular_ = Eigen::Vector3d::Zero();
    double prev_timestamp_ = 0.0;

    // step 9: pose export -- path to rewrite via writePoseFile(); see the loop
    // closure doc comment above for why this is a full-rewrite, not an append.
    std::string pose_out_path_;

    // step 10: path + error-vs-ground-truth
    nav_msgs::msg::Path path_estimated_;
    std::vector<Eigen::Matrix4d> ground_truth_poses_;
    double sum_sq_trans_error_ = 0.0;
    double sum_rot_error_deg_ = 0.0;
    int error_count_ = 0;

    // Loop closure / global map state.
    std::vector<FrameRecord> frame_records_;
    Eigen::Matrix4d last_keyframe_pose_ = Eigen::Matrix4d::Identity(); // raw world_pose_ at the last keyframe
    int frames_since_last_keyframe_ = 0;

    // --- Parameters loaded from config.yaml (see FishboneNode's constructor) ---
    // How often to actually publish path_estimated_ over the wire (still appended
    // every frame regardless -- see publishPaths).
    size_t path_publish_stride_ = 10;
    // Replenish once tracked features drop below this fraction of the last detect() baseline.
    double min_tracked_ratio_ = 0.5;
    // Merge-in dedup: skip fresh detections within this many px of an existing track.
    float merge_min_separation_px_ = 10.0f;

    // --- Keyframe / loop-closure / pose-graph tunables (config.yaml: loop_closure.*) ---
    bool loop_closure_enabled_ = true;
    double keyframe_min_translation_m_ = 5.0;  // new keyframe once the raw chain has moved this far...
    int keyframe_max_frames_ = 15;             // ...or this many frames have passed, whichever comes first
    double odom_edge_weight_ = 5.0;            // pose-graph edge weight for consecutive-keyframe VO (trusted more: dense, frequent)
    double loop_edge_weight_ = 1.0;            // pose-graph edge weight for a verified loop-closure match
    int pose_graph_max_iterations_ = 20;
    // Full pose-file rewrite cadence, in frames (also always done right after a
    // loop-closure optimize(), regardless of this stride -- see maybeAddKeyframeAndCloseLoop).
    size_t pose_file_rewrite_stride_ = 50;

    // Lightweight perf profiling, logged every 30 frames.
    int frame_count_for_timing_ = 0;
    double sum_ms_foveate_ = 0.0, sum_ms_match_ = 0.0, sum_ms_vis_ = 0.0, sum_ms_pose_ = 0.0, sum_ms_total_ = 0.0;
};

} // namespace FISHBONE

#endif // FISHBONE_NODES_FISHBONE_NODE_HPP_
