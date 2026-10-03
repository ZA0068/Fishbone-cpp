#ifndef FISHBONE_VSLAM_POSE_GRAPH_HPP_
#define FISHBONE_VSLAM_POSE_GRAPH_HPP_

#include <Eigen/Dense>
#include <memory>

namespace g2o { class SparseOptimizer; }

namespace FISHBONE {

// Thin wrapper around g2o's SE3 pose-graph optimizer (VertexSE3/EdgeSE3,
// Levenberg-Marquardt). Nodes are keyframe poses in the T_wc convention used
// everywhere else in this codebase (world_pose_: camera-to-world, i.e.
// X_world = pose * X_camera). Edges are relative-pose measurements in the
// SAME T_rel convention Soft2MotionEstimator already returns: for an edge
// from->to, T_rel maps TO's frame into FROM's frame (X_from = T_rel * X_to).
// That happens to be exactly g2o's own EdgeSE3 error convention
// (z^-1 * (x_from^-1 * x_to)), so no extra inversion bookkeeping is needed
// when wiring in odometry (T_rel already produced frame-to-frame every
// callback) or loop-closure edges (ransacKabsch's R,t use the identical
// curr-onto-prev convention).
class PoseGraph {
public:
    PoseGraph();
    ~PoseGraph();

    // Adds a new SE3 node initialized to `pose`. Node 0 is held fixed as the
    // graph's gauge (an SE3 pose graph is only observable up to a rigid
    // transform of the whole graph; anchoring node 0 removes that ambiguity).
    // Returns the new node's index (sequential from 0).
    int addNode(const Eigen::Matrix4d& pose);

    // Adds an edge FROM node `from` TO node `to` with measurement T_rel (see
    // class doc comment for the convention). `weight` scales the edge's
    // information matrix (isotropic 6-DOF); higher = more trusted.
    void addEdge(int from, int to, const Eigen::Matrix4d& T_rel, double weight = 1.0);

    // Runs Levenberg-Marquardt for up to max_iterations. Returns false (no-op)
    // if there are fewer than 2 nodes or no edges yet.
    bool optimize(int max_iterations = 20);

    // Current (possibly optimized) pose of node i.
    Eigen::Matrix4d pose(int i) const;

    int numNodes() const;

private:
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
};

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_POSE_GRAPH_HPP_
