#include <fishbone/vslam/fovean.hpp>

namespace FISHBONE {

Fovean::Fovean(int img_width, int img_height, const FoveanConfig& config)
    : img_width_(img_width), img_height_(img_height), config_(config)
{
}

void Fovean::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat* out_foveated)
{
    // Foveation (center-biased detection + graduated blur) is retired: it
    // structurally concentrated tracked features near the image center --
    // exactly the direction of travel / epipole for a forward-driving car,
    // the single worst place for motion-estimation conditioning (least
    // parallax, most collinear rays). A KITTI run with foveation disabled
    // (sigmas: [0.0]) measured noticeably better early-sequence tracking than
    // the foveated baseline, confirming this. Detection is now uniform GFTT
    // (Shi-Tomasi) across the whole image -- the right choice for a KLT-style
    // pipeline (LK optical flow tracks raw-pixel patches around each keypoint,
    // so a descriptor isn't needed here; GFTT gives sub-pixel-quality, well-
    // ranked corners, which is what LK tracking most benefits from). Spatial
    // spread is enforced by grid bucketing over the whole frame, not a
    // hand-drawn center rect.
    keypoints.clear();

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(img, pts, config_.gftt_max_corners, config_.gftt_quality_level, config_.gftt_min_distance);

    std::vector<cv::KeyPoint> kps;
    kps.reserve(pts.size());
    for (const auto& pt : pts) {
        cv::KeyPoint kp;
        kp.pt = pt;
        kp.response = 100.0f; // GFTT doesn't expose its corner score via this API; bucketing just needs a tie-breaker
        kps.push_back(kp);
    }

    perform_bucketing(kps, keypoints, config_.bucket_size);

    if (out_foveated) {
        *out_foveated = img;
    }
}

cv::Mat Fovean::build_foveated_image(const cv::Mat& img)
{
    // Square (box/Chebyshev) zones instead of circular/elliptical ones. The
    // lens itself is circular, so a smooth radial falloff buys nothing extra
    // once you're this far downstream of it -- and a box test only needs
    // per-axis min/max (no sqrt, no per-pixel mask array, no float32 upcast
    // of the whole image).
    //
    // N sigmas (config_.sigmas) divide the image into N+1 equal concentric box
    // zones; the innermost is always sharp. sigmas is ordered outermost-first,
    // so zone k counting inward from the center (k=1..N, k=0 is the sharp
    // center) uses sigmas[N-k]. See FoveanConfig's doc comment for examples.
    const std::vector<double>& sigmas = config_.sigmas;
    const int n = static_cast<int>(sigmas.size());
    if (n <= 0) {
        return img.clone(); // empty vector: no foveation at all
    }

    int cx = img.cols / 2;
    int cy = img.rows / 2;
    const int total_zones = n + 1; // n blur zones (from the vector) + 1 always-sharp center
    cv::Rect bounds(0, 0, img.cols, img.rows);

    // zone_rect[k]'s outer boundary sits at box fraction (k+1)/total_zones of the
    // half-width/half-height; zone_rect[total_zones-1] == bounds (the full image).
    std::vector<cv::Rect> zone_rect(total_zones);
    for (int k = 0; k < total_zones; ++k) {
        double frac = static_cast<double>(k + 1) / total_zones;
        int hw = static_cast<int>(cx * frac);
        int hh = static_cast<int>(cy * frac);
        zone_rect[k] = cv::Rect(cx - hw, cy - hh, 2 * hw, 2 * hh) & bounds;
    }

    // Paint outermost zone first as the base layer, then progressively overwrite
    // smaller inner rects with lighter (or sharp) zones working inward.
    cv::Mat foveated;
    for (int k = total_zones - 1; k >= 1; --k) {
        double sigma = sigmas[n - k];
        cv::Mat zone_img;
        if (std::isfinite(sigma) && sigma > 0.0) {
            cv::GaussianBlur(img, zone_img, cv::Size(0, 0), sigma);
        } else {
            zone_img = img; // 0/negative/non-finite: "no sigma" for this ring
        }
        if (k == total_zones - 1) {
            foveated = zone_img.clone();
        } else {
            zone_img(zone_rect[k]).copyTo(foveated(zone_rect[k]));
        }
    }
    img(zone_rect[0]).copyTo(foveated(zone_rect[0])); // center: always sharp, explicitly
    return foveated;
}

void Fovean::perform_bucketing(const std::vector<cv::KeyPoint>& input_pts,
                                std::vector<cv::KeyPoint>& output_pts, double bucket_size)
{
    output_pts.clear();
    if (input_pts.empty()) return;
    if (!(bucket_size > 0.0)) { output_pts = input_pts; return; }

    // Grid sized to THIS call's bucket_size, not a fixed member -- see detect()'s
    // call sites: fovea and periphery buckets are deliberately different sizes.
    int cols = std::max(1, static_cast<int>(std::ceil(img_width_ / bucket_size)));
    int rows = std::max(1, static_cast<int>(std::ceil(img_height_ / bucket_size)));

    std::vector<int> bucket_best_idx(static_cast<size_t>(rows) * cols, -1);
    std::vector<float> bucket_best_response(static_cast<size_t>(rows) * cols, -1.0f);

    for (size_t i = 0; i < input_pts.size(); ++i) {
        const auto& kp = input_pts[i];

        int c = static_cast<int>(kp.pt.x / bucket_size);
        int r = static_cast<int>(kp.pt.y / bucket_size);

        if (c >= cols || r >= rows || c < 0 || r < 0) continue;

        int b_idx = r * cols + c;

        // Keep the feature with the strongest response (sharpest corner) in this cell
        if (kp.response > bucket_best_response[b_idx]) {
            bucket_best_response[b_idx] = kp.response;
            bucket_best_idx[b_idx] = static_cast<int>(i);
        }
    }

    // Collect the winners
    for (int idx : bucket_best_idx) {
        if (idx != -1) {
            output_pts.push_back(input_pts[idx]);
        }
    }
}

cv::Point3d Fovean::get_feature_point_3d(const cv::Point2f& pt, double f, double cx, double cy, double b, const cv::Mat& depth_map)
{
    // 1. Get Depth Z from the map (Lidar or Stereo Disparity)
    // Note: Assuming depth_map is CV_32F (meters). If CV_16U (mm), divide by 1000.0.
    double Z = 0.0;

    int u = std::round(pt.x);
    int v = std::round(pt.y);

    // Bounds check
    if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
        Z = static_cast<double>(depth_map.at<float>(v, u));
    }

    // If Depth is invalid (0 or infinity), we cannot project.
    // Return a point with Z=0 or NaN to indicate failure.
    if (Z <= 0.1 || std::isinf(Z) || std::isnan(Z)) {
        return cv::Point3d(0, 0, 0);
    }

    // 2. Reproject to 3D Space (Pinhole Model)
    // X = (u - cx) * Z / f
    // Y = (v - cy) * Z / f
    double X = (pt.x - cx) * Z / f;
    double Y = (pt.y - cy) * Z / f;

    return cv::Point3d(X, Y, Z);
}

} // namespace FISHBONE
