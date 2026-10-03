// File: src/nodes/fishbone_node.cpp

#include <fishbone/nodes/fishbone_node.hpp>
#include <opencv2/highgui.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Geometry>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <chrono>

namespace FISHBONE {

namespace {
constexpr const char* kWindowName = "Fishbone Features";

// Stereo triangulation: pixel disparity -> 3D point in the left-camera frame.
cv::Point3d triangulate(const cv::Point2f& pt_l, const cv::Point2f& pt_r,
                         double fx, double cx, double cy, double baseline)
{
    double d = pt_l.x - pt_r.x;
    if (d < 0.5) return cv::Point3d(0, 0, -1.0); // degenerate/negative disparity
    double Z = (fx * baseline) / d;
    double X = (pt_l.x - cx) * Z / fx;
    double Y = (pt_l.y - cy) * Z / fx;
    return cv::Point3d(X, Y, Z);
}

double median(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// world_pose_ accumulates via repeated matrix multiplication (world_pose_ =
// world_pose_ * T_rel, every frame, thousands of times over a full sequence).
// Floating-point error compounds each time, and rotation matrices have no
// self-correcting structure the way e.g. a normalized quaternion does -- left
// unchecked, the rotation block slowly drifts off SO(3) (stops being exactly
// orthonormal), which introduces shear/scale into every subsequent frame's
// transform and corrupts translation too (evo_rpe's --pose_relation angle_deg
// threw "matrix is not a valid SO(3) group element" after a full run,
// confirming this was actually happening, not just theoretical). Fix: project
// the rotation block back onto the nearest valid rotation via SVD every frame
// (R = U*S*V^T -> R_ortho = U*V^T; same reflection-vs-rotation fix as the
// Kabsch fit in rigid_fit.cpp, since a raw U*V^T can come out as a reflection
// instead of a rotation). This is cheap (one 3x3 SVD) and keeps world_pose_
// exactly on the manifold at every step instead of letting error accumulate.
void orthonormalizeRotation(Eigen::Matrix4d& T)
{
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_ortho = U * V.transpose();
    if (R_ortho.determinant() < 0.0) {
        V.col(2) *= -1.0;
        R_ortho = U * V.transpose();
    }
    T.block<3, 3>(0, 0) = R_ortho;
}
} // namespace

FishboneNode::FishboneNode() : Node("fishbone_node"), has_prev_frame_(false)
{
    world_pose_ = Eigen::Matrix4d::Identity();

    this->declare_parameter("sequence", "00");
    std::string sequence = this->get_parameter("sequence").as_string();

    // --- Load every tunable from config.yaml (or its declared defaults, if the
    // launch file wasn't given a config file) rather than hardcoding them. ---
    int image_width = this->declare_parameter("image_width", 1241);
    int image_height = this->declare_parameter("image_height", 376);

    FishboneConfig fb_config;
    fb_config.fovean.bucket_size = this->declare_parameter("foveation.bucket_size", 50);
    // Foveation blur is retired (uniform GFTT now, see Fovean::detect's doc
    // comment) -- default empty means disabled; still overridable via config.yaml.
    fb_config.fovean.sigmas = this->declare_parameter("foveation.sigmas", std::vector<double>{});
    fb_config.fovean.gftt_max_corners = this->declare_parameter("foveation.gftt_max_corners", 1500);
    fb_config.fovean.gftt_quality_level = this->declare_parameter("foveation.gftt_quality_level", 0.01);
    fb_config.fovean.gftt_min_distance = this->declare_parameter("foveation.gftt_min_distance", 10.0);
    fb_config.max_accel_px = static_cast<float>(this->declare_parameter("matching.max_accel_px", 4.0));
    fb_config.subpixel_patch_half_size = this->declare_parameter("matching.subpixel_patch_half_size", 3);

    min_tracked_ratio_ = this->declare_parameter("matching.min_tracked_ratio", 0.5);
    merge_min_separation_px_ = static_cast<float>(
        this->declare_parameter("matching.merge_min_separation_px", 10.0));

    Soft2MotionConfig motion_config;
    motion_config.ransac_iterations = this->declare_parameter("motion_estimation.ransac_iterations", 300);
    motion_config.ransac_inlier_threshold_m =
        this->declare_parameter("motion_estimation.ransac_inlier_threshold_m", 0.3);
    motion_config.min_points = this->declare_parameter("motion_estimation.min_points", 6);
    motion_config.max_plausible_translation_m =
        this->declare_parameter("motion_estimation.max_plausible_translation_m", 5.0);
    motion_config.min_disparity_px = this->declare_parameter("motion_estimation.min_disparity_px", 2.0);
    motion_config.max_depth_m = this->declare_parameter("motion_estimation.max_depth_m", 50.0);
    motion_config.min_point_spread_m = this->declare_parameter("motion_estimation.min_point_spread_m", 1.0);
    motion_config.max_coast_frames = this->declare_parameter("motion_estimation.max_coast_frames", 10);

    path_publish_stride_ = static_cast<size_t>(
        std::max<int64_t>(1, this->declare_parameter("output.path_publish_stride", 10)));

    loop_closure_enabled_ = this->declare_parameter("loop_closure.enabled", true);
    keyframe_min_translation_m_ = this->declare_parameter("loop_closure.keyframe_min_translation_m", 5.0);
    keyframe_max_frames_ = this->declare_parameter("loop_closure.keyframe_max_frames", 15);
    odom_edge_weight_ = this->declare_parameter("loop_closure.odom_edge_weight", 5.0);
    loop_edge_weight_ = this->declare_parameter("loop_closure.loop_edge_weight", 1.0);
    pose_graph_max_iterations_ = this->declare_parameter("loop_closure.pose_graph_max_iterations", 20);
    pose_file_rewrite_stride_ = static_cast<size_t>(
        std::max<int64_t>(1, this->declare_parameter("output.pose_file_rewrite_stride", 50)));

    LoopClosureConfig lc_config;
    lc_config.akaze_threshold = this->declare_parameter("loop_closure.akaze_threshold", 0.001);
    lc_config.candidate_radius_m = this->declare_parameter("loop_closure.candidate_radius_m", 15.0);
    lc_config.min_keyframe_gap = this->declare_parameter("loop_closure.min_keyframe_gap", 30);
    lc_config.min_keyframes_between_loops =
        this->declare_parameter("loop_closure.min_keyframes_between_loops", 5);
    lc_config.match_ratio_test =
        static_cast<float>(this->declare_parameter("loop_closure.match_ratio_test", 0.75));
    lc_config.ransac_iterations = this->declare_parameter("loop_closure.ransac_iterations", 500);
    lc_config.ransac_inlier_threshold_m =
        this->declare_parameter("loop_closure.ransac_inlier_threshold_m", 0.5);
    lc_config.min_inliers = this->declare_parameter("loop_closure.min_inliers", 15);
    lc_config.min_disparity_px = this->declare_parameter("loop_closure.min_disparity_px", 2.0);
    lc_config.max_depth_m = this->declare_parameter("loop_closure.max_depth_m", 50.0);

    fishbone_ = std::make_unique<Fishbone>(image_width, image_height, fb_config);
    estimator_ = std::make_unique<Soft2MotionEstimator>(motion_config);
    loop_closure_ = std::make_unique<LoopClosureDetector>(lc_config);
    pose_graph_ = std::make_unique<PoseGraph>();

    pose_pub_ = this->create_publisher<fishbone::msg::PoseData>("/fishbone/pose", 10);
    feature_image_pub_ = image_transport::create_publisher(this, "/fishbone/features_image");
    path_estimated_pub_ = this->create_publisher<nav_msgs::msg::Path>("/fishbone/path_estimated", 10);
    path_ground_truth_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "/fishbone/path_ground_truth", rclcpp::QoS(1).transient_local());

    // --- step 9: pose export (see writePoseFile() -- full rewrite, not append,
    // so a loop closure can retroactively correct already-passed frames) ---
    std::filesystem::create_directories("/workspace/generated");
    pose_out_path_ = "/workspace/generated/" + sequence + ".txt";
    RCLCPP_INFO(this->get_logger(), "Writing estimated poses to: %s", pose_out_path_.c_str());

    // --- step 10: ground truth for path + error metrics ---
    std::string gt_path = "/workspace/data/dataset/poses/" + sequence + ".txt";
    ground_truth_poses_ = loadGroundTruthPoses(gt_path);
    if (ground_truth_poses_.empty()) {
        RCLCPP_WARN(this->get_logger(), "No ground truth poses loaded from %s -- error metrics disabled.", gt_path.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(), "Loaded %zu ground truth poses from %s", ground_truth_poses_.size(), gt_path.c_str());

        nav_msgs::msg::Path gt_path_msg;
        gt_path_msg.header.frame_id = "map";
        gt_path_msg.header.stamp = this->now();
        for (const auto& T : ground_truth_poses_) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = gt_path_msg.header;
            ps.pose.position.x = T(0, 3);
            ps.pose.position.y = T(1, 3);
            ps.pose.position.z = T(2, 3);
            Eigen::Quaterniond q(T.block<3, 3>(0, 0));
            ps.pose.orientation.x = q.x();
            ps.pose.orientation.y = q.y();
            ps.pose.orientation.z = q.z();
            ps.pose.orientation.w = q.w();
            gt_path_msg.poses.push_back(ps);
        }
        path_ground_truth_pub_->publish(gt_path_msg);
    }
    path_estimated_.header.frame_id = "map";

    // The live debug window is a nice-to-have, not core functionality -- pose
    // estimation shouldn't die because no X display/auth is available in this
    // session. Disable it gracefully instead of letting an uncaught cv::Exception
    // take the whole node down.
    try {
        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
        gui_available_ = true;
    } catch (const cv::Exception& e) {
        RCLCPP_WARN(this->get_logger(), "No display available, disabling live feature window: %s", e.what());
        gui_available_ = false;
    }

    frame_sub_ = this->create_subscription<fishbone::msg::Frame>(
        "/fishbone/frame", 10,
        std::bind(&FishboneNode::frameCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Fishbone node started, listening on /fishbone/frame");
}

FishboneNode::~FishboneNode()
{
    if (gui_available_) {
        try { cv::destroyWindow(kWindowName); } catch (const cv::Exception&) {}
    }
    // Final flush: writePoseFile() only otherwise runs every pose_file_rewrite_stride_
    // frames (plus after each loop closure), so guarantee the tail isn't lost.
    writePoseFile();
}

std::vector<Eigen::Matrix4d> FishboneNode::loadGroundTruthPoses(const std::string& path) const
{
    std::vector<Eigen::Matrix4d> poses;
    std::ifstream file(path);
    if (!file.is_open()) return poses;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::array<double, 12> v{};
        for (auto& x : v) {
            if (!(ss >> x)) return poses; // malformed line -- bail with what we have
        }
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                T(r, c) = v[r * 4 + c];
            }
        }
        poses.push_back(T);
    }
    return poses;
}

Eigen::Matrix4d FishboneNode::correctedPose(const FrameRecord& rec) const
{
    if (rec.keyframe_idx < 0 || !pose_graph_) return rec.T_local; // shouldn't happen post-bootstrap; be safe anyway
    return pose_graph_->pose(rec.keyframe_idx) * rec.T_local;
}

void FishboneNode::recordFramePose(uint64_t frame_id)
{
    int kf_idx = (loop_closure_ && loop_closure_->size() > 0) ? loop_closure_->size() - 1 : -1;
    Eigen::Matrix4d T_local = Eigen::Matrix4d::Identity();
    if (kf_idx >= 0) {
        // Both sides of this are the raw/uncorrected chain -- see the class doc
        // comment on FrameRecord for why that makes T_local stable forever once written.
        T_local = loop_closure_->keyframe(kf_idx).raw_pose.inverse() * world_pose_;
    }
    frame_records_.push_back({frame_id, kf_idx, T_local});
}

void FishboneNode::maybeAddKeyframeAndCloseLoop(const cv::Mat& raw_left, const cv::Mat& raw_right,
    uint64_t frame_id, double fx, double fy, double cx, double cy, double baseline)
{
    frames_since_last_keyframe_++;
    double dist_since_kf = (world_pose_.block<3, 1>(0, 3) - last_keyframe_pose_.block<3, 1>(0, 3)).norm();
    if (dist_since_kf < keyframe_min_translation_m_ && frames_since_last_keyframe_ < keyframe_max_frames_) {
        return; // not time for a new keyframe yet
    }

    auto t_kf0 = std::chrono::steady_clock::now();
    int kf_idx = loop_closure_->addKeyframe(raw_left, raw_right, world_pose_, frame_id, fx, fy, cx, cy, baseline);
    auto t_kf1 = std::chrono::steady_clock::now();
    int node_idx = pose_graph_->addNode(world_pose_);
    (void)node_idx; // == kf_idx by construction: both grow by exactly one per call, in lockstep

    double ms_addkf = std::chrono::duration<double, std::milli>(t_kf1 - t_kf0).count();
    if (ms_addkf > 200.0) {
        RCLCPP_WARN(this->get_logger(), "[Perf] addKeyframe(%d) took %.0fms (frame %lu)",
            kf_idx, ms_addkf, static_cast<unsigned long>(frame_id));
    }

    if (kf_idx > 0) {
        // Odometry edge from the previous keyframe -- a RAW measurement (both
        // sides come from the never-corrected world_pose_ chain), exactly like
        // every edge in a pose graph is meant to be; only the solved NODE
        // estimates move during optimize(), never the edge measurements.
        const Eigen::Matrix4d& prev_raw = loop_closure_->keyframe(kf_idx - 1).raw_pose;
        Eigen::Matrix4d T_odom = prev_raw.inverse() * world_pose_;
        pose_graph_->addEdge(kf_idx - 1, kf_idx, T_odom, odom_edge_weight_);
    }

    last_keyframe_pose_ = world_pose_;
    frames_since_last_keyframe_ = 0;

    if (kf_idx % 20 == 0) {
        const auto& ls = loop_closure_->stats();
        RCLCPP_INFO(this->get_logger(), "[Keyframe] %d (frame %lu): %zu ORB points with valid depth, pos=(%.1f,%.1f,%.1f)",
            kf_idx, static_cast<unsigned long>(frame_id), loop_closure_->keyframe(kf_idx).points_local.size(),
            world_pose_(0, 3), world_pose_(1, 3), world_pose_(2, 3));
        RCLCPP_INFO(this->get_logger(),
            "[LoopClosure stats] attempts=%ld too_early=%ld own_no_desc=%ld no_candidate_in_range=%ld "
            "candidate_no_desc=%ld too_few_matches=%ld ransac_failed=%ld accepted=%ld",
            ls.attempts, ls.skipped_too_early, ls.skipped_no_own_descriptors, ls.no_candidate_in_range,
            ls.candidate_had_no_descriptors, ls.verification_failed_too_few_matches,
            ls.verification_failed_ransac, ls.accepted);
    }

    if (!loop_closure_enabled_) return;

    auto t_lc0 = std::chrono::steady_clock::now();
    auto lookup = [this](int i) { return pose_graph_->pose(i); };
    auto edge = loop_closure_->tryFindLoopClosure(kf_idx, lookup);
    auto t_lc1 = std::chrono::steady_clock::now();
    double ms_lc = std::chrono::duration<double, std::milli>(t_lc1 - t_lc0).count();
    if (ms_lc > 200.0) {
        RCLCPP_WARN(this->get_logger(), "[Perf] tryFindLoopClosure(%d) took %.0fms (frame %lu)",
            kf_idx, ms_lc, static_cast<unsigned long>(frame_id));
    }
    if (!edge) return;

    pose_graph_->addEdge(edge->from, edge->to, edge->T_rel, loop_edge_weight_);

    Eigen::Matrix4d before = pose_graph_->pose(edge->to);
    pose_graph_->optimize(pose_graph_max_iterations_);
    Eigen::Matrix4d after = pose_graph_->pose(edge->to);
    double correction_m = (after.block<3, 1>(0, 3) - before.block<3, 1>(0, 3)).norm();

    RCLCPP_INFO(this->get_logger(),
        "[LoopClosure] keyframe %d (frame %lu) <-> keyframe %d (frame %lu), %d inliers -- "
        "pose graph optimized, correction at closure point = %.2fm",
        edge->from, static_cast<unsigned long>(loop_closure_->keyframe(edge->from).frame_id),
        edge->to, static_cast<unsigned long>(frame_id), edge->inliers, correction_m);

    // Every already-recorded frame's stored/published pose is derived from the
    // keyframe nodes we just moved -- rewrite both outputs so the correction is
    // actually visible, not just held in pose_graph_'s internal state.
    writePoseFile();
    republishCorrectedPath();
}

void FishboneNode::writePoseFile()
{
    if (pose_out_path_.empty()) return;
    std::ofstream f(pose_out_path_, std::ios::out | std::ios::trunc);
    if (!f.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to (re)write pose output file: %s", pose_out_path_.c_str());
        return;
    }
    for (const auto& rec : frame_records_) {
        Eigen::Matrix4d T = correctedPose(rec);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                f << T(r, c);
                if (!(r == 2 && c == 3)) f << " ";
            }
        }
        f << "\n";
    }
}

void FishboneNode::republishCorrectedPath()
{
    nav_msgs::msg::Path corrected;
    corrected.header.frame_id = "map";
    corrected.header.stamp = this->now();
    corrected.poses.reserve(frame_records_.size());
    for (const auto& rec : frame_records_) {
        Eigen::Matrix4d T = correctedPose(rec);
        geometry_msgs::msg::PoseStamped ps;
        ps.header = corrected.header;
        ps.pose.position.x = T(0, 3);
        ps.pose.position.y = T(1, 3);
        ps.pose.position.z = T(2, 3);
        Eigen::Quaterniond q(T.block<3, 3>(0, 0));
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();
        corrected.poses.push_back(ps);
    }
    // Replace, not append: subsequent incremental publishPaths() calls now build
    // on top of this corrected baseline until the next loop closure rebuilds it again.
    path_estimated_ = corrected;
    path_estimated_pub_->publish(path_estimated_);
}

void FishboneNode::publishPaths(const rclcpp::Time& stamp)
{
    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id = "map";
    ps.header.stamp = stamp;
    ps.pose.position.x = world_pose_(0, 3);
    ps.pose.position.y = world_pose_(1, 3);
    ps.pose.position.z = world_pose_(2, 3);
    Eigen::Quaterniond q(world_pose_.block<3, 3>(0, 0));
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();

    path_estimated_.header.stamp = stamp;
    path_estimated_.poses.push_back(ps);

    // The whole (ever-growing) poses array gets serialized on every publish -- that's
    // fine early on but turns into real, steadily-growing per-frame cost over a long
    // sequence (this is what "[Perf] pose=..." climbing over time was, empirically:
    // most of it wasn't pose estimation at all, it was this). Append every frame so
    // the path is complete, but only publish periodically.
    if (path_estimated_.poses.size() % path_publish_stride_ == 0) {
        path_estimated_pub_->publish(path_estimated_);
    }
}

void FishboneNode::updateAndLogError(uint64_t frame_id, const Eigen::Matrix4d& T_rel)
{
    if (ground_truth_poses_.empty()) return;
    // frame_id==0 has no predecessor to form a relative motion from; frame_id can also
    // exceed the ground truth size once camera_node loops back to the start of the sequence.
    if (frame_id == 0 || frame_id >= ground_truth_poses_.size()) return;

    const Eigen::Matrix4d& T_gt_prev = ground_truth_poses_[frame_id - 1];
    const Eigen::Matrix4d& T_gt_curr = ground_truth_poses_[frame_id];
    Eigen::Matrix4d T_gt_rel = T_gt_prev.inverse() * T_gt_curr;

    // Relative Pose Error: how far our estimated step deviates from the ground-truth step.
    Eigen::Matrix4d T_err = T_gt_rel.inverse() * T_rel;
    double trans_error = T_err.block<3, 1>(0, 3).norm();

    Eigen::Matrix3d R_err = T_err.block<3, 3>(0, 0);
    double cos_angle = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
    double rot_error_deg = std::acos(cos_angle) * 180.0 / M_PI;

    sum_sq_trans_error_ += trans_error * trans_error;
    sum_rot_error_deg_ += rot_error_deg;
    error_count_++;

    if (error_count_ % 50 == 0) {
        double rmse_trans = std::sqrt(sum_sq_trans_error_ / error_count_);
        double mean_rot = sum_rot_error_deg_ / error_count_;
        RCLCPP_INFO(this->get_logger(),
            "[Error vs ground truth] over %d frames: RPE trans RMSE=%.3f m, mean rot error=%.3f deg",
            error_count_, rmse_trans, mean_rot);
    }
}

void FishboneNode::computeAngularMotion(const Eigen::Matrix4d& T_rel, double dt,
    Eigen::Vector3d& out_vel_ang, Eigen::Vector3d& out_acc_ang)
{
    Eigen::AngleAxisd aa(T_rel.block<3, 3>(0, 0));
    out_vel_ang = aa.axis() * aa.angle() / dt;

    if (has_prev_angular_velocity_) {
        out_acc_ang = (out_vel_ang - prev_velocity_angular_) / dt;
    } else {
        out_acc_ang.setZero();
    }
    prev_velocity_angular_ = out_vel_ang;
    has_prev_angular_velocity_ = true;
}

void FishboneNode::computeLinearMotionFromFeatures(double fx, double cx, double cy, double baseline,
    double dt, Eigen::Vector3d& out_vel_lin, Eigen::Vector3d& out_acc_lin) const
{
    std::vector<double> vx, vy, vz, ax, ay, az;

    for (const auto& f : features_) {
        // "use only detected features for L', L'', R' and R''": skip anything that
        // didn't get the full 3-frame cross-temporal match this round.
        if (!f.has_cross_history) continue;

        cv::Point3d P_t  = triangulate(f.pt,         f.pt_r_curr,  fx, cx, cy, baseline);
        cv::Point3d P_t1 = triangulate(f.pt_l_prev,  f.pt_r_prev,  fx, cx, cy, baseline);
        cv::Point3d P_t2 = triangulate(f.pt_l_prev2, f.pt_r_prev2, fx, cx, cy, baseline);
        if (P_t.z <= 0 || P_t1.z <= 0 || P_t2.z <= 0) continue;
        if (P_t.z > 80.0 || P_t1.z > 80.0 || P_t2.z > 80.0) continue; // same far-depth cutoff as soft2_motion.cpp

        // Ipsi-temporal (L<->L', R<->R'): velocity from the current vs. one-frame-back position.
        cv::Point3d v_ipsi((P_t.x - P_t1.x) / dt, (P_t.y - P_t1.y) / dt, (P_t.z - P_t1.z) / dt);
        // Cross-temporal (L'<->R'', R'<->L''): the older velocity sample, one frame further back.
        cv::Point3d v_cross((P_t1.x - P_t2.x) / dt, (P_t1.y - P_t2.y) / dt, (P_t1.z - P_t2.z) / dt);

        vx.push_back(v_ipsi.x); vy.push_back(v_ipsi.y); vz.push_back(v_ipsi.z);
        ax.push_back((v_ipsi.x - v_cross.x) / dt);
        ay.push_back((v_ipsi.y - v_cross.y) / dt);
        az.push_back((v_ipsi.z - v_cross.z) / dt);
    }

    // Median aggregate across features -- robust to individual mismatches/outliers,
    // consistent with the median scale-recovery already used in soft2_motion.cpp.
    out_vel_lin = Eigen::Vector3d(median(vx), median(vy), median(vz));
    out_acc_lin = Eigen::Vector3d(median(ax), median(ay), median(az));
}

void FishboneNode::publishFeatureVisualization(const cv::Mat& left_img, const rclcpp::Time& stamp)
{
    // Always draw and show locally, regardless of whether anything subscribed to the ROS topic.
    // left_img is the FOVEATED image -- what's actually being tracked on.
    cv::Mat vis;
    cv::cvtColor(left_img, vis, cv::COLOR_GRAY2BGR);

    for (const auto& f : features_) {
        const bool tracked = f.age > 0;
        const cv::Scalar color = tracked ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255); // green=tracked, yellow=new
        cv::circle(vis, f.pt, 3, color, -1);
        if (tracked) {
            // Motion vector from where this feature was last frame to where it is now.
            cv::line(vis, f.pt_l_prev, f.pt, cv::Scalar(255, 0, 0), 1);
        }
    }

    cv::putText(vis, "features: " + std::to_string(features_.size()), cv::Point(10, 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    if (gui_available_) {
        cv::imshow(kWindowName, vis);
        cv::waitKey(1); // Pumps the GUI event loop so the window actually paints/refreshes.
    }

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = "camera_left";
    auto img_msg = cv_bridge::CvImage(header, "bgr8", vis).toImageMsg();
    feature_image_pub_.publish(img_msg);
}

void FishboneNode::frameCallback(const fishbone::msg::Frame::SharedPtr msg)
{
    if (!msg->is_stereo) {
        RCLCPP_WARN_ONCE(this->get_logger(), "Monocular frames are not supported yet, skipping.");
        return;
    }

    // DIAGNOSTIC: confirm/deny frame-drop theory -- circularMatch and the
    // ground-truth RPE comparison both assume consecutive frame_ids arrive
    // back-to-back. If this ever fires, that assumption is false and explains
    // degraded tracking (LK's search window sized for 1-frame motion, seeing
    // N-frame motion instead) independent of any algorithmic issue.
    if (has_prev_frame_ && msg->frame_id != last_seen_frame_id_ + 1) {
        RCLCPP_WARN(this->get_logger(), "[FrameGap] expected frame_id %lu, got %lu (gap of %ld)",
            static_cast<unsigned long>(last_seen_frame_id_ + 1), static_cast<unsigned long>(msg->frame_id),
            static_cast<int64_t>(msg->frame_id) - static_cast<int64_t>(last_seen_frame_id_) - 1);
    }
    last_seen_frame_id_ = msg->frame_id;

    auto t_start = std::chrono::steady_clock::now();

    cv::Mat raw_left = cv_bridge::toCvCopy(msg->stereo_camera.left_camera.image, "mono8")->image;
    cv::Mat raw_right = cv_bridge::toCvCopy(msg->stereo_camera.right_camera.image, "mono8")->image;

    // Foveate BOTH cameras once, here, before anything else touches them. Detection,
    // matching/tracking, sub-pixel refinement, storage as prev_*, and visualization
    // all operate on these foveated images from this point on -- not the raw frames.
    cv::Mat curr_left = fishbone_->foveate(raw_left);
    cv::Mat curr_right = fishbone_->foveate(raw_right);
    auto t_foveate = std::chrono::steady_clock::now();

    // Camera intrinsics/baseline -- needed by both the bootstrap branch (to seed
    // keyframe 0) and the main path (motion estimation), so compute once, up front.
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        msg->stereo_camera.left_camera.k[0], msg->stereo_camera.left_camera.k[1], msg->stereo_camera.left_camera.k[2],
        msg->stereo_camera.left_camera.k[3], msg->stereo_camera.left_camera.k[4], msg->stereo_camera.left_camera.k[5],
        msg->stereo_camera.left_camera.k[6], msg->stereo_camera.left_camera.k[7], msg->stereo_camera.left_camera.k[8]);

    double fx = msg->stereo_camera.left_camera.k[0];
    double fy = msg->stereo_camera.left_camera.k[4];
    double cx = msg->stereo_camera.left_camera.k[2];
    double cy = msg->stereo_camera.left_camera.k[5];
    double baseline = 0.0;
    if (fx > 1e-6) {
        // KITTI: P(0,3) = -fx * Tx, so baseline between cam2 (left) and cam3 (right)
        // is the difference of their Tx offsets from the reference camera.
        baseline = std::abs((msg->stereo_camera.left_camera.p[3] - msg->stereo_camera.right_camera.p[3]) / fx);
    }

    if (!has_prev_frame_) {
        // Bootstrap: nothing to track against yet, just seed the detector.
        fishbone_->detect(curr_left, curr_right, features_);
        baseline_feature_count_ = features_.size();
        RCLCPP_INFO(this->get_logger(), "Bootstrapped tracking with %zu features", features_.size());

        publishFeatureVisualization(curr_left, this->now());

        // Keyframe 0 / pose-graph node 0, anchored at world_pose_ == Identity (no
        // motion has been integrated yet). Every later keyframe chains off this one,
        // and frame 0 itself needs a FrameRecord too so the output pose file has
        // one line per frame_id, matching ground truth's indexing exactly.
        int kf0 = loop_closure_->addKeyframe(raw_left, raw_right, world_pose_, msg->frame_id, fx, fy, cx, cy, baseline);
        pose_graph_->addNode(world_pose_); // == kf0 by construction
        last_keyframe_pose_ = world_pose_;
        frames_since_last_keyframe_ = 0;
        recordFramePose(msg->frame_id);
        (void)kf0;

        prev_left_ = curr_left;
        prev_right_ = curr_right;
        has_prev_frame_ = true;
        return;
    }

    // Reseed tracked features with last iteration's quadratic-predicted position
    // instead of blindly reusing where they were (zero-velocity guess). Freshly
    // detected features (age == 0) have no prediction yet, so leave them as-is.
    for (auto& f : features_) {
        if (f.age > 0) {
            f.pt = f.pred_l;
        }
    }

    // 1. Track previously detected features into the new frame via the quad
    //    (L/R/L'/R'/L''/R'') matching DAG. prev_prev_* are empty until the 3rd
    //    real frame, at which point circularMatch automatically starts using them.
    auto [attempted, tracked] = fishbone_->circularMatch(
        curr_left, curr_right, prev_left_, prev_right_, prev_prev_left_, prev_prev_right_, features_);
    auto t_match = std::chrono::steady_clock::now();

    // 2. Replenish once tracked features drop below 50% of the last detect() baseline.
    //    Interpolate rather than hard-replace: keep whatever's still tracking fine
    //    (and its velocity/acceleration history) and merge in fresh detections that
    //    aren't sitting right on top of an existing track.
    if (static_cast<double>(features_.size()) < min_tracked_ratio_ * static_cast<double>(baseline_feature_count_)) {
        std::vector<FeaturePoint> fresh;
        fishbone_->detect(curr_left, curr_right, fresh);

        size_t merged_in = 0;
        for (const auto& cand : fresh) {
            bool too_close = false;
            for (const auto& existing : features_) {
                float dx = cand.pt.x - existing.pt.x;
                float dy = cand.pt.y - existing.pt.y;
                if (dx * dx + dy * dy < merge_min_separation_px_ * merge_min_separation_px_) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                features_.push_back(cand);
                ++merged_in;
            }
        }
        baseline_feature_count_ = features_.size();
        RCLCPP_INFO(this->get_logger(), "Tracked %d/%d, merged in %zu fresh detections (now %zu, below %.0f%% threshold)",
            tracked, attempted, merged_in, features_.size(), min_tracked_ratio_ * 100.0);
    } else {
        RCLCPP_INFO(this->get_logger(), "Tracked %d/%d features", tracked, attempted);
    }

    publishFeatureVisualization(curr_left, this->now());
    auto t_vis = std::chrono::steady_clock::now();

    // 3. Motion estimation from the tracked features. K/fx/fy/cx/cy/baseline were
    // already computed above (shared with the bootstrap branch).
    // NOTE: The lidar->camera extrinsic ('Tr') isn't available in this sequence's
    // calib.txt (only P0-P3 are present -- confirmed by reading it directly), so we
    // fall back to Identity. Lidar depth lookups will mostly miss as a result, and
    // soft2_motion.cpp will fall back to stereo triangulation for scale recovery instead.
    Eigen::Matrix4d T_cam_lidar = Eigen::Matrix4d::Identity();

    Eigen::Matrix4d T_rel = estimator_->estimate(features_, K, msg->lidar_data.points, T_cam_lidar, baseline);

    // Accumulate: world_pose_ expresses the camera pose in world coordinates, T_rel maps
    // current-camera-frame coordinates into the previous-camera-frame.
    world_pose_ = world_pose_ * T_rel;
    // Repeated multiplication drifts the rotation block off SO(3) over thousands
    // of frames -- project it back onto the nearest valid rotation every step
    // (see orthonormalizeRotation's doc comment).
    orthonormalizeRotation(world_pose_);

    // Temporal + spatial: position (world_pose_ above); angular velocity/acceleration
    // from T_rel's rotation; linear velocity/acceleration from direct feature
    // triangulation (ipsi-temporal for velocity, cross-temporal for acceleration).
    double dt = msg->timestamp - prev_timestamp_;
    if (!(dt > 1e-6)) dt = 0.1; // first call, or a bad timestamp: fall back to the nominal 10 Hz period

    Eigen::Vector3d vel_ang, acc_ang;
    computeAngularMotion(T_rel, dt, vel_ang, acc_ang);

    Eigen::Vector3d vel_lin, acc_lin;
    computeLinearMotionFromFeatures(fx, cx, cy, baseline, dt, vel_lin, acc_lin);

    prev_timestamp_ = msg->timestamp;

    fishbone::msg::PoseData pose_msg;
    pose_msg.frame_id = msg->frame_id;
    pose_msg.timestamp = msg->timestamp;
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(pose_msg.pose.data()) = world_pose_;
    Eigen::Map<Eigen::Vector3d>(pose_msg.velocity_linear.data()) = vel_lin;
    Eigen::Map<Eigen::Vector3d>(pose_msg.velocity_angular.data()) = vel_ang;
    Eigen::Map<Eigen::Vector3d>(pose_msg.acceleration_linear.data()) = acc_lin;
    Eigen::Map<Eigen::Vector3d>(pose_msg.acceleration_angular.data()) = acc_ang;
    pose_pub_->publish(pose_msg);

    // Attribute this frame to its most recent keyframe and record its local
    // (keyframe-relative) pose -- see FrameRecord's doc comment. This is what
    // lets a later loop closure retroactively correct this frame's stored/
    // published pose without needing to re-run VO.
    recordFramePose(msg->frame_id);

    // step 10: live path + running error vs ground truth
    publishPaths(this->now());
    updateAndLogError(msg->frame_id, T_rel);

    // Keyframe / loop-closure / pose-graph bookkeeping -- adds a keyframe once
    // the trigger policy fires, and (if a loop closure is found) optimizes the
    // graph and rewrites the corrected pose file + RViz path in place.
    maybeAddKeyframeAndCloseLoop(raw_left, raw_right, msg->frame_id, fx, fy, cx, cy, baseline);

    // step 9: persist to /workspace/generated/<sequence>.txt -- periodically here;
    // maybeAddKeyframeAndCloseLoop already did it unconditionally above if a loop
    // just closed, and the destructor does a final one at shutdown either way.
    if (frame_records_.size() % pose_file_rewrite_stride_ == 0) writePoseFile();

    // 4. Roll the 3-frame window forward: t-1 becomes t-2, current becomes t-1.
    prev_prev_left_ = prev_left_;
    prev_prev_right_ = prev_right_;
    prev_left_ = curr_left;
    prev_right_ = curr_right;

    auto t_end = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    frame_count_for_timing_++;
    sum_ms_foveate_ += ms(t_start, t_foveate);
    sum_ms_match_ += ms(t_foveate, t_match);
    sum_ms_vis_ += ms(t_match, t_vis);
    sum_ms_pose_ += ms(t_vis, t_end);
    sum_ms_total_ += ms(t_start, t_end);
    if (frame_count_for_timing_ % 30 == 0) {
        RCLCPP_INFO(this->get_logger(),
            "[Perf] avg over %d frames: foveate=%.1fms match=%.1fms vis=%.1fms pose=%.1fms TOTAL=%.1fms (%.1f fps)",
            frame_count_for_timing_,
            sum_ms_foveate_ / frame_count_for_timing_, sum_ms_match_ / frame_count_for_timing_,
            sum_ms_vis_ / frame_count_for_timing_, sum_ms_pose_ / frame_count_for_timing_,
            sum_ms_total_ / frame_count_for_timing_, 1000.0 / (sum_ms_total_ / frame_count_for_timing_));

        const auto& s = estimator_->stats();
        RCLCPP_INFO(this->get_logger(),
            "[Motion] over %ld calls: accepted=%ld (%.0f%%) coasted=%ld (%.0f%%) coast_capped=%ld rejects: "
            "too_few_features=%ld too_few_correspondences=%ld low_point_spread=%ld "
            "too_few_inliers=%ld translation_too_large=%ld exception=%ld",
            s.calls, s.accepted, s.calls > 0 ? 100.0 * s.accepted / s.calls : 0.0,
            s.coasted, s.calls > 0 ? 100.0 * s.coasted / s.calls : 0.0, s.coast_capped,
            s.reject_too_few_features, s.reject_too_few_correspondences, s.reject_low_point_spread,
            s.reject_too_few_inliers, s.reject_translation_too_large, s.reject_exception);
        if (s.reject_translation_too_large > 0) {
            RCLCPP_INFO(this->get_logger(),
                "[Motion] rejected solves: mean translation=%.2fm max=%.2fm mean inliers=%.1f mean correspondences=%.1f "
                "| accepted solves: mean translation=%.2fm mean inliers=%.1f mean correspondences=%.1f",
                s.sum_rejected_translation_m / s.reject_translation_too_large, s.max_rejected_translation_m,
                static_cast<double>(s.sum_rejected_inliers) / s.reject_translation_too_large,
                static_cast<double>(s.sum_rejected_correspondences) / s.reject_translation_too_large,
                s.accepted > 0 ? s.sum_accepted_translation_m / s.accepted : 0.0,
                s.accepted > 0 ? static_cast<double>(s.sum_accepted_inliers) / s.accepted : 0.0,
                s.accepted > 0 ? static_cast<double>(s.sum_accepted_correspondences) / s.accepted : 0.0);
        }
    }
}

} // namespace FISHBONE
