#ifndef FISHBONE_VSLAM_SOFT2_FEATURE_HPP_
#define FISHBONE_VSLAM_SOFT2_FEATURE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <fishbone/vslam/fovean.hpp> // Correct include path

namespace FISHBONE {

struct FeaturePoint {
    cv::Point2f pt;         // Current Left Point (The "Anchor")

    // Trace points for the circular match
    cv::Point2f pt_l_prev;
    cv::Point2f pt_r_prev;
    cv::Point2f pt_r_curr;

    // t-2 positions, derived via the cross-temporal matches (L'->R'', R'->L'').
    // Only meaningful when has_cross_history is true (i.e. this call had 3-frame
    // history available); callers doing physical velocity/acceleration should
    // gate on that flag rather than assume these are populated.
    cv::Point2f pt_l_prev2;
    cv::Point2f pt_r_prev2;
    bool has_cross_history = false;

    // Quadratic-extrapolated position this feature is predicted to occupy next
    // frame (see circularMatch doc). Used as the LK seed on the following call
    // instead of blindly reusing the current position (zero-velocity guess).
    cv::Point2f pred_l;
    cv::Point2f pred_r;

    int age = 0;
    float response = 0.0f;
};

// Tunables for Fishbone's own matching stage, on top of FoveanConfig's
// detection tunables. Defaults match the last hand-tuned values.
struct FishboneConfig {
    FoveanConfig fovean;

    float max_accel_px = 4.0f;          // circularMatch rejection: max plausible px/frame^2
    int subpixel_patch_half_size = 3;   // subpixelRefine SSD patch half-size (7x7 at default)
};

class Fishbone {
public:
    Fishbone(int width, int height, const FishboneConfig& config = FishboneConfig());

    // Main pipeline function
    void detect(const cv::Mat& img_left, const cv::Mat& img_right,
               std::vector<FeaturePoint>& features, cv::Mat* out_foveated = nullptr);

    // Builds the foveated view (sharp center, blurred periphery) for visualization,
    // independent of whether detect()/circularMatch ran this frame.
    cv::Mat foveate(const cv::Mat& img) { return fovean_.build_foveated_image(img); }

    // Three-frame quad match, run as a small concurrent DAG:
    //   step 3: L  <-> R    current stereo                          (needs: seed) -- HIGHEST
    //           PRIORITY: runs alone, first, not raced against anything else, since
    //           this is what position/depth is triangulated from.
    //   step 4: L  <-> L'   and  R  <-> R'   direct/ipsi temporal    (4a needs seed; 4b needs step3)
    //   step 5: L' <-> R''  and  R' <-> L''  cross temporal          (5a needs 4a;   5b needs 4b)
    // step 5 requires img_l_prev2/img_r_prev2 (t-2); pass empty Mats to skip it
    // (e.g. only 1 frame of history available yet) and fall back to steps 3-4 only.
    //
    // Each match is sub-pixel refined (quadratic/parabola fit on the local SSD
    // surface). Where 3-frame history is available, the three time samples per
    // trajectory (t, t-1, t-2) get a constant-velocity (linear) then constant-
    // acceleration (quadratic) fit; features whose implied acceleration is
    // physically implausible are rejected ("deny... as it's not making sense").
    // Accepted features get pred_l/pred_r: the quadratic extrapolation to t+1,
    // used as next call's seed.
    // Returns {attempted, accepted}.
    std::pair<int, int> circularMatch(
        const cv::Mat& img_l_curr, const cv::Mat& img_r_curr,
        const cv::Mat& img_l_prev, const cv::Mat& img_r_prev,
        const cv::Mat& img_l_prev2, const cv::Mat& img_r_prev2,
        std::vector<FeaturePoint>& features);

private:
    int width_, height_;
    FishboneConfig config_;

    // Fovean is now a MEMBER variable so it survives
    Fovean fovean_;

    // Refines target_pt to sub-pixel accuracy by fitting a parabola to the
    // local SSD surface (against ref_img's patch at ref_pt), independently in x and y.
    cv::Point2f subpixelRefine(const cv::Mat& ref_img, const cv::Point2f& ref_pt,
                                const cv::Mat& target_img, const cv::Point2f& target_pt) const;
};

} // namespace FISHBONE

#endif // FISHBONE_VSLAM_SOFT2_FEATURE_HPP_
