// File: include/fishbone/vslam/fovean.hpp

#ifndef FISHBONE_VSLAM_FOVEAN_HPP_
#define FISHBONE_VSLAM_FOVEAN_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace FISHBONE {

// All the tunables that used to be hardcoded magic numbers in fovean.cpp.
// Defaults here match the last hand-tuned values; config.yaml overrides them
// at runtime via fishbone_node without needing a rebuild.
struct FoveanConfig {
    int bucket_size = 50;         // spatial bucketing grid cell size (px), for even feature spread

    // Foveation blur is retired (see Fovean::detect's doc comment: it
    // structurally concentrated tracked features near the image center --
    // the worst place for motion-estimation conditioning in forward-driving
    // footage). build_foveated_image() is kept for anyone who wants to
    // re-enable graduated blur via config without a rebuild, but the default
    // is now "no foveation at all": an empty vector disables it entirely, and
    // that's what detect() itself no longer depends on regardless.
    //
    // N sigmas divide the image into N+1 equal concentric box (Chebyshev)
    // zones; the innermost zone is ALWAYS sharp (sigma=0), regardless of N.
    // sigmas is ordered OUTERMOST-FIRST. A 0, negative, or non-finite entry
    // means "no blur" for that specific ring.
    //   {}         -> whole image sharp (default -- no foveation)
    //   {s}        -> outer 50% = s,                    inner 50% sharp
    //   {s0, s1}   -> outer 33% = s0, middle 33% = s1,   inner 33% sharp
    std::vector<double> sigmas = {};

    // Shi-Tomasi (GoodFeaturesToTrack) parameters, applied uniformly across
    // the whole image -- see detect()'s doc comment for why GFTT+grid-bucketing
    // (not a descriptor-based detector) is the right choice for this KLT-style,
    // LK-optical-flow-tracked pipeline.
    int gftt_max_corners = 1500;          // raised from the old fovea-only 200: now covers the whole frame,
                                           // not a 100x100 crop, so bucketing (not this cap) should be what
                                           // actually governs final spatial density
    double gftt_quality_level = 0.01;
    double gftt_min_distance = 10.0;
};

class Fovean {
public:
    Fovean(int img_width, int img_height, const FoveanConfig& config = FoveanConfig());

    // Uniform GFTT detection + grid bucketing across the whole image. See the
    // .cpp's doc comment for why foveation (center-biased detection) was retired.
    void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat* out_foveated = nullptr);
    cv::Point3d get_feature_point_3d(const cv::Point2f& pt, double f, double cx, double cy, double b, const cv::Mat& depth_map);

    // Builds a graduated-blur view per config_.sigmas. With the default empty
    // sigmas this is a no-op (returns img.clone()) -- kept so blur can still be
    // re-enabled via config.yaml without a rebuild, but detect() no longer
    // depends on its output being anything other than the identity.
    cv::Mat build_foveated_image(const cv::Mat& img);

private:
    int img_width_, img_height_;
    FoveanConfig config_;

    // bucket_size is a parameter (not just the member config_.bucket_size)
    // because callers besides detect() may want a different grid resolution.
    void perform_bucketing(const std::vector<cv::KeyPoint>& input_pts,
                            std::vector<cv::KeyPoint>& output_pts, double bucket_size);
};

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_FOVEAN_HPP_
