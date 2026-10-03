#include <fishbone/vslam/rigid_fit.hpp>
#include <random>

namespace FISHBONE {

bool kabschFit(const std::vector<Eigen::Vector3d>& curr_pts,
               const std::vector<Eigen::Vector3d>& prev_pts,
               Eigen::Matrix3d& out_R, Eigen::Vector3d& out_t)
{
    const size_t n = curr_pts.size();
    if (n < 3 || prev_pts.size() != n) return false;

    Eigen::Vector3d centroid_curr = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid_prev = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < n; ++i) {
        centroid_curr += curr_pts[i];
        centroid_prev += prev_pts[i];
    }
    centroid_curr /= static_cast<double>(n);
    centroid_prev /= static_cast<double>(n);

    // Cross-covariance of the two centered point sets.
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < n; ++i) {
        H += (curr_pts[i] - centroid_curr) * (prev_pts[i] - centroid_prev).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d& U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0.0) {
        // Reflection instead of a rotation -- standard Kabsch fix: flip the sign
        // of V's last column (corresponding to the smallest singular value) and
        // recompute.
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }

    if (!R.allFinite()) return false;

    out_R = R;
    out_t = centroid_prev - R * centroid_curr;
    return true;
}

RigidFitResult ransacKabsch(const std::vector<Eigen::Vector3d>& curr_pts,
                             const std::vector<Eigen::Vector3d>& prev_pts,
                             int iterations, double inlier_threshold_m,
                             int min_inliers, unsigned rng_seed)
{
    RigidFitResult result;
    const size_t n = curr_pts.size();
    if (n < 3 || prev_pts.size() != n) return result;

    std::mt19937 rng(rng_seed);
    std::uniform_int_distribution<size_t> pick(0, n - 1);

    int best_inlier_count = -1;
    std::vector<int> best_inliers;
    const double thresh_sq = inlier_threshold_m * inlier_threshold_m;

    for (int iter = 0; iter < iterations; ++iter) {
        size_t i0 = pick(rng), i1 = pick(rng), i2 = pick(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        std::vector<Eigen::Vector3d> sc{curr_pts[i0], curr_pts[i1], curr_pts[i2]};
        std::vector<Eigen::Vector3d> sp{prev_pts[i0], prev_pts[i1], prev_pts[i2]};
        if (!kabschFit(sc, sp, R, t)) continue;

        std::vector<int> inliers;
        inliers.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            double err_sq = (R * curr_pts[i] + t - prev_pts[i]).squaredNorm();
            if (err_sq < thresh_sq) inliers.push_back(static_cast<int>(i));
        }
        if (static_cast<int>(inliers.size()) > best_inlier_count) {
            best_inlier_count = static_cast<int>(inliers.size());
            best_inliers = std::move(inliers);
        }
    }

    if (best_inlier_count < min_inliers) return result; // result.ok stays false

    std::vector<Eigen::Vector3d> in_curr, in_prev;
    in_curr.reserve(best_inliers.size());
    in_prev.reserve(best_inliers.size());
    for (int idx : best_inliers) {
        in_curr.push_back(curr_pts[idx]);
        in_prev.push_back(prev_pts[idx]);
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    if (!kabschFit(in_curr, in_prev, R, t)) return result;

    result.ok = true;
    result.R = R;
    result.t = t;
    result.inliers = std::move(best_inliers);
    return result;
}

} // namespace FISHBONE
