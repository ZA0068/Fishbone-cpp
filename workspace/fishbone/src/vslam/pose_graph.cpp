#include <fishbone/vslam/pose_graph.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

namespace FISHBONE {

namespace {

g2o::Isometry3 toIsometry(const Eigen::Matrix4d& T)
{
    g2o::Isometry3 iso = g2o::Isometry3::Identity();
    iso.linear() = T.block<3, 3>(0, 0);
    iso.translation() = T.block<3, 1>(0, 3);
    return iso;
}

Eigen::Matrix4d fromIsometry(const g2o::Isometry3& iso)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = iso.rotation();
    T.block<3, 1>(0, 3) = iso.translation();
    return T;
}

} // namespace

PoseGraph::PoseGraph()
{
    optimizer_ = std::make_unique<g2o::SparseOptimizer>();

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linear_solver = std::make_unique<LinearSolverType>();
    auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));
    auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    optimizer_->setAlgorithm(algorithm);
    optimizer_->setVerbose(false);
}

PoseGraph::~PoseGraph() = default;

int PoseGraph::addNode(const Eigen::Matrix4d& pose)
{
    int id = static_cast<int>(optimizer_->vertices().size());
    auto* v = new g2o::VertexSE3();
    v->setId(id);
    v->setEstimate(toIsometry(pose));
    v->setFixed(id == 0); // anchor the gauge on the very first node
    optimizer_->addVertex(v);
    return id;
}

void PoseGraph::addEdge(int from, int to, const Eigen::Matrix4d& T_rel, double weight)
{
    auto* e = new g2o::EdgeSE3();
    e->setVertex(0, optimizer_->vertex(from));
    e->setVertex(1, optimizer_->vertex(to));
    e->setMeasurement(toIsometry(T_rel));

    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity() * weight;
    e->setInformation(information);

    // Loop-closure edges in particular can occasionally be a bad match that
    // still passed RANSAC's inlier count by chance; a robust kernel caps how
    // much a single bad edge can drag the whole graph instead of letting it
    // dominate the quadratic loss.
    auto* kernel = new g2o::RobustKernelHuber();
    e->setRobustKernel(kernel);

    optimizer_->addEdge(e);
}

bool PoseGraph::optimize(int max_iterations)
{
    if (optimizer_->vertices().size() < 2 || optimizer_->edges().empty()) return false;
    optimizer_->initializeOptimization();
    optimizer_->optimize(max_iterations);
    return true;
}

Eigen::Matrix4d PoseGraph::pose(int i) const
{
    auto* v = dynamic_cast<g2o::VertexSE3*>(optimizer_->vertex(i));
    if (!v) return Eigen::Matrix4d::Identity();
    return fromIsometry(v->estimate());
}

int PoseGraph::numNodes() const
{
    return static_cast<int>(optimizer_->vertices().size());
}

} // namespace FISHBONE
