// INCLUDE THE HEADER THAT DEFINES THE CLASS 'FISHBONE'
#include <fishbone/vslam/fishbone.hpp> 
#include <opencv2/video/tracking.hpp> 
#include <algorithm>
#include <limits>
#include <future>

namespace FISHBONE {

Fishbone::Fishbone(int width, int height, const FishboneConfig& config)
    : width_(width), height_(height), config_(config),
      fovean_(width, height, config.fovean)
{
}

void Fishbone::detect(const cv::Mat& img_left, const cv::Mat& img_right, 
                      std::vector<FeaturePoint>& features, cv::Mat* out_foveated) 
{
    (void)img_right; // Silence unused parameter warning

    std::vector<cv::KeyPoint> kps;
    fovean_.detect(img_left, kps, out_foveated);

    features.clear();
    features.reserve(kps.size());

    for (const auto& kp : kps) {
        FeaturePoint fp;
        fp.pt = kp.pt;
        fp.response = kp.response;
        fp.age = 0;
        features.push_back(fp);
    }
}

cv::Point2f Fishbone::subpixelRefine(const cv::Mat& ref_img, const cv::Point2f& ref_pt,
                                      const cv::Mat& target_img, const cv::Point2f& target_pt) const
{
    const int half = config_.subpixel_patch_half_size;
    const int patch_dim = 2 * half + 1;

    cv::Rect ref_rect(static_cast<int>(std::round(ref_pt.x)) - half,
                       static_cast<int>(std::round(ref_pt.y)) - half,
                       patch_dim, patch_dim);
    if ((ref_rect & cv::Rect(0, 0, ref_img.cols, ref_img.rows)) != ref_rect) {
        return target_pt; // Too close to the border to safely refine.
    }
    cv::Mat ref_patch = ref_img(ref_rect);

    const double kInvalid = std::numeric_limits<double>::max();
    auto ssdAt = [&](int ox, int oy) -> double {
        cv::Rect r(static_cast<int>(std::round(target_pt.x)) + ox - half,
                   static_cast<int>(std::round(target_pt.y)) + oy - half,
                   patch_dim, patch_dim);
        if ((r & cv::Rect(0, 0, target_img.cols, target_img.rows)) != r) return kInvalid;

        cv::Mat diff;
        cv::absdiff(ref_patch, target_img(r), diff);
        diff.convertTo(diff, CV_32F);
        return cv::sum(diff.mul(diff))[0];
    };

    double e_center = ssdAt(0, 0);
    double e_xm = ssdAt(-1, 0), e_xp = ssdAt(1, 0);
    double e_ym = ssdAt(0, -1), e_yp = ssdAt(0, 1);

    float dx = 0.0f, dy = 0.0f;

    if (e_xm < kInvalid && e_xp < kInvalid && e_center < kInvalid) {
        double denom = e_xm - 2.0 * e_center + e_xp;
        if (std::abs(denom) > 1e-6) {
            dx = static_cast<float>(0.5 * (e_xm - e_xp) / denom);
        }
    }
    if (e_ym < kInvalid && e_yp < kInvalid && e_center < kInvalid) {
        double denom = e_ym - 2.0 * e_center + e_yp;
        if (std::abs(denom) > 1e-6) {
            dy = static_cast<float>(0.5 * (e_ym - e_yp) / denom);
        }
    }

    // A parabola fit should only nudge sub-pixel; clamp to guard against a bad fit.
    dx = std::clamp(dx, -1.0f, 1.0f);
    dy = std::clamp(dy, -1.0f, 1.0f);

    return cv::Point2f(target_pt.x + dx, target_pt.y + dy);
}

namespace {

struct LKResult {
    std::vector<cv::Point2f> pts;
    std::vector<uchar> status;
};

LKResult runLK(const cv::Mat& img_from, const cv::Mat& img_to, const std::vector<cv::Point2f>& seed) {
    LKResult r;
    std::vector<float> err;
    cv::Size winSize(21, 21);
    int maxLevel = 3;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img_from, img_to, seed, r.pts, r.status, err, winSize, maxLevel, criteria);
    return r;
}

// Constant-velocity (linear) then constant-acceleration (quadratic) fit through
// three equally time-spaced samples x2 (t-2), x1 (t-1), x0 (t). Returns the
// implied acceleration magnitude (for sanity-rejection) and the quadratic
// extrapolation to t+1 (for use as the next call's seed).
struct TemporalFit {
    cv::Point2f predicted_next;
    float accel_mag = 0.0f;
};

TemporalFit fitTemporal(const cv::Point2f& x0, const cv::Point2f& x1, const cv::Point2f& x2) {
    cv::Point2f v = x1 - x2;         // step 6: linear/velocity, from the two oldest samples
    cv::Point2f a = (x0 - x1) - v;   // step 7: quadratic/acceleration, folding in the newest sample
    TemporalFit fit;
    fit.accel_mag = std::sqrt(a.x * a.x + a.y * a.y);
    fit.predicted_next = x0 + v + a; // extrapolate one step further
    return fit;
}

} // namespace

std::pair<int, int> Fishbone::circularMatch(
    const cv::Mat& img_l_curr, const cv::Mat& img_r_curr,
    const cv::Mat& img_l_prev, const cv::Mat& img_r_prev,
    const cv::Mat& img_l_prev2, const cv::Mat& img_r_prev2,
    std::vector<FeaturePoint>& features) 
{
    if (features.empty()) return {0, 0};

    const bool have_prev2 = !img_l_prev2.empty() && !img_r_prev2.empty();
    const float kMaxAccelPx = config_.max_accel_px; // px/frame^2 -- reject implausible acceleration

    std::vector<cv::Point2f> pts_l;
    pts_l.reserve(features.size());
    for (const auto& f : features) pts_l.push_back(f.pt);

    // --- Wave 1 (highest priority): step 3, L-R stereo. This is what depth/position
    // gets triangulated from, so it runs alone -- not raced against step 4a for
    // CPU/thread scheduling the way the previous version did. ---
    LKResult step3 = runLK(img_l_curr, img_r_curr, pts_l);

    // --- Wave 2: step 4a (L-L', ipsi) and step 4b (R-R', ipsi, needs step3 for its seed) ---
    auto fut_step4a = std::async(std::launch::async, runLK, std::cref(img_l_curr), std::cref(img_l_prev), std::cref(pts_l));
    auto fut_step4b = std::async(std::launch::async, runLK, std::cref(img_r_curr), std::cref(img_r_prev), std::cref(step3.pts));
    LKResult step4a = fut_step4a.get();
    LKResult step4b = fut_step4b.get();

    // --- Wave 3: step 5a (L'-R'', cross, needs step4a) and step 5b (R'-L'', cross, needs step4b) ---
    LKResult step5a, step5b;
    if (have_prev2) {
        auto fut_step5a = std::async(std::launch::async, runLK, std::cref(img_l_prev), std::cref(img_r_prev2), std::cref(step4a.pts));
        auto fut_step5b = std::async(std::launch::async, runLK, std::cref(img_r_prev), std::cref(img_l_prev2), std::cref(step4b.pts));
        step5a = fut_step5a.get();
        step5b = fut_step5b.get();
    }

    std::vector<FeaturePoint> valid_features;
    valid_features.reserve(features.size());
    int valid_count = 0;

    for (size_t i = 0; i < features.size(); ++i) {
        if (!step3.status[i] || !step4a.status[i] || !step4b.status[i]) continue;
        if (have_prev2 && (!step5a.status[i] || !step5b.status[i])) continue;

        // Sub-pixel refine each raw LK match against the patch it was tracked from.
        cv::Point2f r_curr = subpixelRefine(img_l_curr, pts_l[i],      img_r_curr, step3.pts[i]);
        cv::Point2f l_prev = subpixelRefine(img_l_curr, pts_l[i],      img_l_prev, step4a.pts[i]);
        cv::Point2f r_prev = subpixelRefine(img_r_curr, step3.pts[i],  img_r_prev, step4b.pts[i]);

        cv::Point2f pred_l = pts_l[i]; // fallback: zero-velocity guess when we lack 3-frame history
        cv::Point2f pred_r = r_curr;
        cv::Point2f r_prev2, l_prev2; // t-2 samples, via cross-temporal (only valid if have_prev2)

        if (have_prev2) {
            // Step 5's outputs land in the OPPOSITE camera's t-2 image (L'-R'' feeds the
            // R-trajectory's t-2 sample; R'-L'' feeds the L-trajectory's), so each trajectory
            // still gets an independent, cross-camera-verified history sample.
            r_prev2 = subpixelRefine(img_l_prev, l_prev, img_r_prev2, step5a.pts[i]);
            l_prev2 = subpixelRefine(img_r_prev, r_prev, img_l_prev2, step5b.pts[i]);

            TemporalFit fit_l = fitTemporal(pts_l[i], l_prev, l_prev2);
            TemporalFit fit_r = fitTemporal(r_curr,   r_prev, r_prev2);

            // "deny and reject it as it's not making sense": implausible acceleration
            // means the chain disagrees with itself across time -> not the same point.
            if (fit_l.accel_mag > kMaxAccelPx || fit_r.accel_mag > kMaxAccelPx) continue;

            pred_l = fit_l.predicted_next;
            pred_r = fit_r.predicted_next;
        }

        FeaturePoint fp = features[i];
        fp.pt_r_curr = r_curr;
        fp.pt_l_prev = l_prev;
        fp.pt_r_prev = r_prev;
        fp.pred_l = pred_l;
        fp.pred_r = pred_r;
        fp.has_cross_history = have_prev2;
        if (have_prev2) {
            fp.pt_r_prev2 = r_prev2;
            fp.pt_l_prev2 = l_prev2;
        }
        fp.age++;
        valid_features.push_back(fp);
        valid_count++;
    }

    features = valid_features;
    return { (int)pts_l.size(), valid_count };
}

} // namespace FISHBONE
