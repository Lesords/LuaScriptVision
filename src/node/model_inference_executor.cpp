#include "model_inference_executor.h"
#include "model_node_utils.h"
#include "resource_limits.h"

#include "modules/cv/frame.h"

#ifdef USE_CVI_TPU
#include "inference/cvi_session.h"
#include "inference/tpu_scheduler.h"
#endif

#ifdef USE_CVI_MPI
#include "modules/cv/cv_helpers.h"
#include "modules/cv/cv_types.h"
#include "modules/cv/cvi_vpss_processor.h"
#endif

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace node {
namespace {

constexpr float kArcFaceReferenceWidth = 112.0f;
constexpr float kArcFaceReferenceHeight = 112.0f;
const std::array<cv::Point2f, 5> kArcFaceReferencePoints = {{
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f},
}};

// Limits concurrent VPSS preprocess + TPU inference to match VB pool capacity.
// Pool 3 has VB_POOL3_TOTAL blocks; exceeding this causes NOBUF (0xc006800e).
class VpssSlotSemaphore {
public:
    explicit VpssSlotSemaphore(int max_count) : count_(max_count) {}

    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_ > 0; });
        --count_;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_one();
    }

private:
    int count_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// RAII guard for VpssSlotSemaphore
struct VpssSlotGuard {
    VpssSlotSemaphore& sem;
    explicit VpssSlotGuard(VpssSlotSemaphore& s) : sem(s) { sem.acquire(); }
    ~VpssSlotGuard() { sem.release(); }
    VpssSlotGuard(const VpssSlotGuard&) = delete;
    VpssSlotGuard& operator=(const VpssSlotGuard&) = delete;
};

VpssSlotSemaphore& vpss_slot_semaphore() {
    static VpssSlotSemaphore sem(VB_POOL3_TOTAL);
    return sem;
}

void fill_resize_meta(int source_w,
                      int source_h,
                      int target_w,
                      int target_h,
                      PreprocessMeta* meta) {
    meta->scale_x = static_cast<float>(target_w) /
                    static_cast<float>(std::max(1, source_w));
    meta->scale_y = static_cast<float>(target_h) /
                    static_cast<float>(std::max(1, source_h));
    meta->scale = meta->scale_x;
    meta->pad_x = 0;
    meta->pad_y = 0;
    meta->ori_w = source_w;
    meta->ori_h = source_h;
    meta->input_w = target_w;
    meta->input_h = target_h;
}

std::array<cv::Point2f, 5> scaled_arcface_reference_points(int target_w, int target_h) {
    const float scale_x = static_cast<float>(target_w) / kArcFaceReferenceWidth;
    const float scale_y = static_cast<float>(target_h) / kArcFaceReferenceHeight;
    std::array<cv::Point2f, 5> points = kArcFaceReferencePoints;
    for (auto& point : points) {
        point.x *= scale_x;
        point.y *= scale_y;
    }
    return points;
}

cv::Mat align_face(const cv::Mat& source,
                   const SelectedRoi& roi,
                   int target_w,
                   int target_h,
                   int fill_value) {
    if (source.empty() || !roi.has_keypoints) {
        return cv::Mat();
    }

    // getAffineTransform requires exactly 3 point pairs.
    // Use left eye, right eye, nose tip (indices 0, 1, 2).
    std::vector<cv::Point2f> source_points = {
        cv::Point2f(roi.keypoints[0].x, roi.keypoints[0].y),
        cv::Point2f(roi.keypoints[1].x, roi.keypoints[1].y),
        cv::Point2f(roi.keypoints[2].x, roi.keypoints[2].y),
    };

    auto reference = scaled_arcface_reference_points(target_w, target_h);
    std::vector<cv::Point2f> target_points = {reference[0], reference[1], reference[2]};
    // Use getAffineTransform (imgproc) instead of estimateAffinePartial2D (calib3d)
    // since the cross-compile OpenCV may not include calib3d module.
    // Both produce the same 2x3 affine matrix when given exactly 3 point pairs.
    cv::Mat affine = cv::getAffineTransform(source_points, target_points);
    if (affine.empty()) {
        return cv::Mat();
    }

    cv::Mat aligned;
    cv::warpAffine(source,
                   aligned,
                   affine,
                   cv::Size(target_w, target_h),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(fill_value, fill_value, fill_value));
    return aligned;
}

#ifdef USE_CVI_TPU
void copy_run_stats(const inference::CviSession::RunStats& stats,
                    double* input_ms,
                    double* forward_ms,
                    double* output_ms) {
    *input_ms = stats.input_ms;
    *forward_ms = stats.forward_ms;
    *output_ms = stats.output_ms;
}
#endif

}  // namespace

FullFrameExecutionResult execute_full_frame_inference(const lua_cv::Frame& frame,
                                                      const ModelExecutorConfig& config) {
    FullFrameExecutionResult result;
    const PreprocessConfig& preprocess = *config.preprocess_config;
    auto t_pre_start = std::chrono::steady_clock::now();

    int target_w = preprocess.input_width > 0 ? preprocess.input_width : frame.width();
    int target_h = preprocess.input_height > 0 ? preprocess.input_height : frame.height();

    result.preprocess_meta.ori_w = frame.width();
    result.preprocess_meta.ori_h = frame.height();
    result.preprocess_meta.input_w = target_w;
    result.preprocess_meta.input_h = target_h;

#ifdef USE_CVI_TPU
    if (!config.session) {
        throw std::runtime_error("CviSession not initialized");
    }

    bool use_vb = false;
    lua_cv::Frame preprocessed;
    std::string preprocess_type = to_lower(preprocess.type);

#ifdef USE_CVI_MPI
    // Limit concurrent VPSS preprocess to available VB pool blocks.
    // Without this, 3 ModelNodes can exhaust Pool 3 (2 blocks) causing NOBUF.
    VpssSlotGuard vpss_guard(vpss_slot_semaphore());

    if (config.session->supports_vb_input() &&
        frame.storage_type() == lua_cv::Frame::StorageType::CVI) {
        auto spec = config.session->get_vb_input_spec();
        lua_cv::PixelFormat out_pf = lua_cv::from_cvi_pixel_format(spec.pixel_format);
        target_w = static_cast<int>(spec.width);
        target_h = static_cast<int>(spec.height);
        result.preprocess_meta.input_w = target_w;
        result.preprocess_meta.input_h = target_h;

        try {
            result.timings.vpss_attempted = true;
            auto t_vpss_start = std::chrono::steady_clock::now();
            lua_cv::Frame work;
            if (frame.video_frame()) {
                work = lua_cv::Frame(*frame.video_frame(), false);
            } else {
                work = frame.clone();
            }

            lua_cv::CviVpssProcessor local_vpss;
            lua_cv::CviVpssProcessor& vpss = config.vpss_processor ? *config.vpss_processor : local_vpss;
            if (preprocess_type == "letterbox") {
                result.preprocess_meta = compute_letterbox_meta(frame.width(), frame.height(),
                                                                target_w, target_h,
                                                                preprocess.center);
                vpss.letterbox(work, target_w, target_h,
                               static_cast<uint8_t>(preprocess.fill_value),
                               nullptr, out_pf);
            } else if (preprocess_type == "resize_center_crop") {
                float r = std::max(static_cast<float>(target_w) / frame.width(),
                                   static_cast<float>(target_h) / frame.height());
                int rw = static_cast<int>(std::ceil(frame.width() * r));
                int rh = static_cast<int>(std::ceil(frame.height() * r));
                int cx = (rw - target_w) / 2;
                int cy = (rh - target_h) / 2;
                vpss.crop_resize(work, cx, cy, target_w, target_h, target_w, target_h, out_pf);
                result.preprocess_meta.scale_x = static_cast<float>(target_w) /
                    static_cast<float>(std::max(1, frame.width()));
                result.preprocess_meta.scale_y = static_cast<float>(target_h) /
                    static_cast<float>(std::max(1, frame.height()));
                result.preprocess_meta.scale = result.preprocess_meta.scale_x;
                result.preprocess_meta.ori_w = frame.width();
                result.preprocess_meta.ori_h = frame.height();
                result.preprocess_meta.input_w = target_w;
                result.preprocess_meta.input_h = target_h;
                result.preprocess_meta.pad_x = 0;
                result.preprocess_meta.pad_y = 0;
            } else if (preprocess_type == "resize" || preprocess_type == "none") {
                if (frame.width() != target_w || frame.height() != target_h ||
                    preprocess_type == "resize") {
                    vpss.resize(work, target_w, target_h);
                }
                fill_resize_meta(frame.width(), frame.height(), target_w, target_h,
                                 &result.preprocess_meta);
                if (work.pixel_format() != out_pf) {
                    vpss.convert_format(work, out_pf);
                }
            } else {
                throw std::runtime_error("Unsupported preprocess type: " + preprocess.type);
            }

            preprocessed = std::move(work);
            std::string reason;
            if (lua_cv::cv_helpers::can_zero_copy(
                    preprocessed,
                    spec.pixel_format,
                    spec.width,
                    spec.height,
                    &reason)) {
                use_vb = true;
            }
            result.timings.preprocess_path = use_vb ? "vpss_vb" : "vpss_copy";
            result.timings.vpss_ms = elapsed_ms(t_vpss_start, std::chrono::steady_clock::now());
        } catch (const std::exception& e) {
            result.warning = ExecutionWarning{"VPSS preprocess failed", e.what()};
            preprocessed = lua_cv::Frame();
            // For CVI VB-input models (INT8), the CPU float fallback cannot work
            // because the model expects quantized VB buffers, not float data.
            // Re-throw to surface the VPSS failure rather than producing a misleading
            // shape-mismatch error downstream.
            if (config.session->supports_vb_input()) {
                throw;
            }
        }
    }
#endif

    if (use_vb) {
#ifdef USE_CVI_MPI
        auto vb_mem = preprocessed.as_vb_memory();
        if (!vb_mem) {
            throw std::runtime_error("Failed to get VB memory from frame");
        }

        auto t_pre_end = std::chrono::steady_clock::now();
        result.timings.preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
#ifdef USE_CVI_TPU
        auto tpu_result = inference::TpuScheduler::instance().submit_vb(
            config.session, vb_mem->physical_addr(), vb_mem->size_bytes());
        // Release VPSS output frame immediately after TPU inference completes.
        // Returns VB block to pool before function scope ends.
        preprocessed = lua_cv::Frame();
        if (!tpu_result.success) {
            throw std::runtime_error("TPU inference failed: " + tpu_result.error);
        }
        result.outputs = std::move(tpu_result.outputs);
        result.output_shapes = std::move(tpu_result.output_shapes);
#else
        config.session->run_vb(vb_mem, &result.outputs, &result.output_shapes);
        preprocessed = lua_cv::Frame();
#endif
        auto t_infer_end = std::chrono::steady_clock::now();
        result.timings.infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        copy_run_stats(config.session->last_run_stats(),
                       &result.timings.tpu_input_ms,
                       &result.timings.tpu_forward_ms,
                       &result.timings.tpu_output_ms);
        result.timings.use_vb = true;
#endif
    } else if (config.session->supports_vb_input()) {
        // INT8 VB-input model but frame is not in CVI storage (e.g. cached CPU clone).
        // Cannot use CPU float fallback - the quantized model expects VB input.
        // Skip this frame rather than crashing in run_all().
        result.skipped = true;
    } else {
        cv::Mat mat;
        bool skip_preprocess = false;
        if (!preprocessed.empty()) {
            mat = preprocessed.to_mat_copy();
            skip_preprocess = true;
        } else {
            mat = frame.to_mat_copy();
        }
        if (mat.empty()) {
            throw std::runtime_error("Frame is empty");
        }

        auto t_cpu_start = std::chrono::steady_clock::now();
        if (!skip_preprocess && preprocess_type == "letterbox") {
            if (target_w <= 0 || target_h <= 0) {
                target_w = mat.cols;
                target_h = mat.rows;
            }
            result.preprocess_meta = compute_letterbox_meta(mat.cols, mat.rows, target_w, target_h,
                                                            preprocess.center);

            int new_w = static_cast<int>(std::floor(mat.cols * result.preprocess_meta.scale));
            int new_h = static_cast<int>(std::floor(mat.rows * result.preprocess_meta.scale));
            cv::Mat resized;
            if (new_w > 0 && new_h > 0 &&
                (new_w != mat.cols || new_h != mat.rows)) {
                cv::resize(mat, resized, cv::Size(new_w, new_h));
            } else {
                resized = mat;
            }

            int pad_w = target_w - new_w;
            int pad_h = target_h - new_h;
            int left = preprocess.center ? pad_w / 2 : 0;
            int top = preprocess.center ? pad_h / 2 : 0;
            int right = std::max(0, pad_w - left);
            int bottom = std::max(0, pad_h - top);

            cv::copyMakeBorder(resized, mat, top, bottom, left, right,
                               cv::BORDER_CONSTANT,
                               cv::Scalar(preprocess.fill_value,
                                          preprocess.fill_value,
                                          preprocess.fill_value));
        } else if (!skip_preprocess && preprocess_type == "resize_center_crop") {
            float r = std::max(static_cast<float>(target_w) / mat.cols,
                               static_cast<float>(target_h) / mat.rows);
            int rw = static_cast<int>(std::ceil(mat.cols * r));
            int rh = static_cast<int>(std::ceil(mat.rows * r));
            cv::resize(mat, mat, cv::Size(rw, rh));
            int cx = (rw - target_w) / 2;
            int cy = (rh - target_h) / 2;
            mat = mat(cv::Rect(cx, cy, target_w, target_h)).clone();
            fill_resize_meta(frame.width(), frame.height(), target_w, target_h,
                             &result.preprocess_meta);
        } else if (!skip_preprocess && (preprocess_type == "resize" || preprocess_type == "none")) {
            if (target_w > 0 && target_h > 0 &&
                (mat.cols != target_w || mat.rows != target_h || preprocess_type == "resize")) {
                cv::resize(mat, mat, cv::Size(target_w, target_h));
            }
            fill_resize_meta(frame.width(), frame.height(), target_w, target_h,
                             &result.preprocess_meta);
        } else if (!skip_preprocess) {
            throw std::runtime_error("Unsupported preprocess type: " + preprocess.type);
        }
        result.timings.cpu_pre_ms = elapsed_ms(t_cpu_start, std::chrono::steady_clock::now());

        std::vector<float> input_data;
        std::vector<int64_t> input_shape;
        auto t_build_start = std::chrono::steady_clock::now();
        build_float_input(mat, preprocess, &input_data, &input_shape);
        result.timings.build_input_ms = elapsed_ms(t_build_start, std::chrono::steady_clock::now());

        auto t_pre_end = std::chrono::steady_clock::now();
        result.timings.preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
#ifdef USE_CVI_TPU
        auto tpu_result = inference::TpuScheduler::instance().submit_float(
            config.session, input_data.data(), input_data.size(), input_shape);
        if (!tpu_result.success) {
            throw std::runtime_error("TPU inference failed: " + tpu_result.error);
        }
        result.outputs = std::move(tpu_result.outputs);
        result.output_shapes = std::move(tpu_result.output_shapes);
#else
        config.session->run_all(
            input_data.data(),
            input_shape,
            &result.outputs,
            &result.output_shapes);
#endif
        auto t_infer_end = std::chrono::steady_clock::now();
        result.timings.infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        copy_run_stats(config.session->last_run_stats(),
                       &result.timings.tpu_input_ms,
                       &result.timings.tpu_forward_ms,
                       &result.timings.tpu_output_ms);
    }
#else
    (void)frame;
    (void)config;
#endif

    return result;
}

RoiExecutionResult execute_roi_inference(const lua_cv::Frame& frame,
                                         const SelectedRoi& roi,
                                         const ModelExecutorConfig& config) {
    RoiExecutionResult result;
    const PreprocessConfig& preprocess = *config.preprocess_config;
    auto t_pre_start = std::chrono::steady_clock::now();

    int target_w = config.crop_size_explicit ? config.crop_width : preprocess.input_width;
    int target_h = config.crop_size_explicit ? config.crop_height : preprocess.input_height;
    if (target_w <= 0 || target_h <= 0) {
        target_w = roi.roi.w;
        target_h = roi.roi.h;
    }

    result.preprocess_meta.ori_w = roi.roi.w;
    result.preprocess_meta.ori_h = roi.roi.h;
    result.preprocess_meta.input_w = target_w;
    result.preprocess_meta.input_h = target_h;

#ifdef USE_CVI_TPU
    if (!config.session) {
        throw std::runtime_error("CviSession not initialized");
    }

    bool use_vb = false;
    lua_cv::Frame preprocessed;
    std::string preprocess_type = to_lower(preprocess.type);

#ifdef USE_CVI_MPI
    // Limit concurrent VPSS preprocess to available VB pool blocks.
    VpssSlotGuard vpss_guard(vpss_slot_semaphore());

    if (preprocess_type != "face_align" &&
        config.session->supports_vb_input() &&
        frame.storage_type() == lua_cv::Frame::StorageType::CVI) {
        auto spec = config.session->get_vb_input_spec();
        lua_cv::PixelFormat out_pf = lua_cv::from_cvi_pixel_format(spec.pixel_format);
        target_w = static_cast<int>(spec.width);
        target_h = static_cast<int>(spec.height);
        result.preprocess_meta.input_w = target_w;
        result.preprocess_meta.input_h = target_h;

        try {
            result.vpss_attempted = true;
            auto t_vpss_start = std::chrono::steady_clock::now();
            lua_cv::Frame work;
            if (frame.video_frame()) {
                work = lua_cv::Frame(*frame.video_frame(), false);
            } else {
                work = frame.clone();
            }

            lua_cv::CviVpssProcessor local_vpss;
            lua_cv::CviVpssProcessor& vpss = config.vpss_processor ? *config.vpss_processor : local_vpss;
            if (preprocess_type == "letterbox") {
                vpss.crop(work, roi.roi.x, roi.roi.y, roi.roi.w, roi.roi.h);
                result.preprocess_meta = compute_letterbox_meta(roi.roi.w, roi.roi.h, target_w, target_h,
                                                                preprocess.center);
                vpss.letterbox(work, target_w, target_h,
                               static_cast<uint8_t>(preprocess.fill_value),
                               nullptr, out_pf);
            } else if (preprocess_type == "resize" || preprocess_type == "none") {
                if (roi.roi.w != target_w || roi.roi.h != target_h || preprocess_type == "resize") {
                    vpss.crop_resize(work, roi.roi.x, roi.roi.y, roi.roi.w, roi.roi.h,
                                     target_w, target_h, out_pf);
                } else {
                    vpss.crop(work, roi.roi.x, roi.roi.y, roi.roi.w, roi.roi.h);
                    if (work.pixel_format() != out_pf) {
                        vpss.convert_format(work, out_pf);
                    }
                }
                fill_resize_meta(roi.roi.w, roi.roi.h, target_w, target_h, &result.preprocess_meta);
            } else if (preprocess_type == "resize_center_crop") {
                float r = std::max(static_cast<float>(target_w) / roi.roi.w,
                                   static_cast<float>(target_h) / roi.roi.h);
                int rw = static_cast<int>(std::ceil(roi.roi.w * r));
                int rh = static_cast<int>(std::ceil(roi.roi.h * r));
                int cx = (rw - target_w) / 2;
                int cy = (rh - target_h) / 2;
                vpss.crop_resize(work, roi.roi.x + cx, roi.roi.y + cy, target_w, target_h,
                                 target_w, target_h, out_pf);
                result.preprocess_meta.scale_x = static_cast<float>(target_w) /
                    static_cast<float>(std::max(1, roi.roi.w));
                result.preprocess_meta.scale_y = static_cast<float>(target_h) /
                    static_cast<float>(std::max(1, roi.roi.h));
                result.preprocess_meta.scale = result.preprocess_meta.scale_x;
                result.preprocess_meta.ori_w = roi.roi.w;
                result.preprocess_meta.ori_h = roi.roi.h;
                result.preprocess_meta.input_w = target_w;
                result.preprocess_meta.input_h = target_h;
                result.preprocess_meta.pad_x = 0;
                result.preprocess_meta.pad_y = 0;
            } else {
                throw std::runtime_error("Unsupported preprocess type: " + preprocess.type);
            }

            preprocessed = std::move(work);
            std::string reason;
            if (lua_cv::cv_helpers::can_zero_copy(
                    preprocessed,
                    spec.pixel_format,
                    spec.width,
                    spec.height,
                    &reason)) {
                use_vb = true;
            }
            result.use_vb = use_vb;
            result.vpss_ms = elapsed_ms(t_vpss_start, std::chrono::steady_clock::now());
        } catch (const std::exception& e) {
            result.warning = ExecutionWarning{"VPSS ROI preprocess failed", e.what()};
            preprocessed = lua_cv::Frame();
        }
    }
#endif

    if (use_vb) {
#ifdef USE_CVI_MPI
        auto vb_mem = preprocessed.as_vb_memory();
        if (!vb_mem) {
            throw std::runtime_error("Failed to get VB memory from ROI frame");
        }
        auto t_pre_end = std::chrono::steady_clock::now();
        result.preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
#ifdef USE_CVI_TPU
        auto tpu_result = inference::TpuScheduler::instance().submit_vb(
            config.session, vb_mem->physical_addr(), vb_mem->size_bytes());
        if (!tpu_result.success) {
            throw std::runtime_error("TPU inference failed: " + tpu_result.error);
        }
        result.outputs = std::move(tpu_result.outputs);
        result.output_shapes = std::move(tpu_result.output_shapes);
#else
        config.session->run_vb(vb_mem, &result.outputs, &result.output_shapes);
#endif
        auto t_infer_end = std::chrono::steady_clock::now();
        result.infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        copy_run_stats(config.session->last_run_stats(),
                       &result.tpu_input_ms,
                       &result.tpu_forward_ms,
                       &result.tpu_output_ms);
#endif
    } else {
        cv::Mat mat;
        cv::Mat src;
        bool skip_preprocess = false;
        if (!preprocessed.empty()) {
            mat = preprocessed.to_mat_copy();
            skip_preprocess = true;
        } else {
            src = frame.to_mat_copy();
            if (src.empty()) {
                result.valid = false;
                return result;
            }

            if (preprocess_type != "face_align") {
                cv::Rect roi_rect(roi.roi.x, roi.roi.y, roi.roi.w, roi.roi.h);
                mat = src(roi_rect).clone();
            }
        }

        auto t_cpu_start = std::chrono::steady_clock::now();
        if (!skip_preprocess && preprocess_type == "letterbox") {
            result.preprocess_meta = compute_letterbox_meta(roi.roi.w, roi.roi.h, target_w, target_h,
                                                            preprocess.center);
            int new_w = static_cast<int>(std::floor(roi.roi.w * result.preprocess_meta.scale));
            int new_h = static_cast<int>(std::floor(roi.roi.h * result.preprocess_meta.scale));
            cv::Mat resized;
            if (new_w > 0 && new_h > 0 &&
                (new_w != roi.roi.w || new_h != roi.roi.h)) {
                cv::resize(mat, resized, cv::Size(new_w, new_h));
            } else {
                resized = mat;
            }

            int pad_w = target_w - new_w;
            int pad_h = target_h - new_h;
            int left = preprocess.center ? pad_w / 2 : 0;
            int top = preprocess.center ? pad_h / 2 : 0;
            int right = std::max(0, pad_w - left);
            int bottom = std::max(0, pad_h - top);

            cv::copyMakeBorder(resized, mat, top, bottom, left, right,
                               cv::BORDER_CONSTANT,
                               cv::Scalar(preprocess.fill_value,
                                          preprocess.fill_value,
                                          preprocess.fill_value));
        } else if (!skip_preprocess && preprocess_type == "resize_center_crop") {
            float r = std::max(static_cast<float>(target_w) / mat.cols,
                               static_cast<float>(target_h) / mat.rows);
            int rw = static_cast<int>(std::ceil(mat.cols * r));
            int rh = static_cast<int>(std::ceil(mat.rows * r));
            cv::resize(mat, mat, cv::Size(rw, rh));
            int cx = (rw - target_w) / 2;
            int cy = (rh - target_h) / 2;
            mat = mat(cv::Rect(cx, cy, target_w, target_h)).clone();
            fill_resize_meta(roi.roi.w, roi.roi.h, target_w, target_h, &result.preprocess_meta);
        } else if (!skip_preprocess && preprocess_type == "face_align") {
            if (!roi.has_keypoints) {
                result.warning = ExecutionWarning{"Face alignment skipped", "ROI is missing 5-point landmarks"};
                result.valid = false;
                return result;
            }
            mat = align_face(src, roi, target_w, target_h, preprocess.fill_value);
            if (mat.empty()) {
                result.warning = ExecutionWarning{"Face alignment failed", "Failed to estimate affine transform"};
                result.valid = false;
                return result;
            }
            fill_resize_meta(roi.roi.w, roi.roi.h, target_w, target_h, &result.preprocess_meta);
            if (preprocess.save_crop) {
                static int face_debug_idx = 0;
                cv::imwrite(preprocess.save_path + "/face_align_" + std::to_string(face_debug_idx++) + ".jpg", mat);
            }
        } else if (!skip_preprocess &&
                   (preprocess_type == "resize" || preprocess_type == "none")) {
            if (mat.cols != target_w || mat.rows != target_h || preprocess_type == "resize") {
                cv::resize(mat, mat, cv::Size(target_w, target_h));
            }
            fill_resize_meta(roi.roi.w, roi.roi.h, target_w, target_h, &result.preprocess_meta);
            if (preprocess.save_crop) {
                static int roi_debug_idx = 0;
                cv::imwrite(preprocess.save_path + "/roi_crop_" + std::to_string(roi_debug_idx++) + ".jpg", mat);
            }
        } else if (!skip_preprocess) {
            throw std::runtime_error("Unsupported preprocess type: " + preprocess.type);
        }
        result.cpu_pre_ms = elapsed_ms(t_cpu_start, std::chrono::steady_clock::now());

        std::vector<float> input_data;
        std::vector<int64_t> input_shape;
        auto t_build_start = std::chrono::steady_clock::now();
        build_float_input(mat, preprocess, &input_data, &input_shape);
        result.build_input_ms = elapsed_ms(t_build_start, std::chrono::steady_clock::now());

        auto t_pre_end = std::chrono::steady_clock::now();
        result.preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
#ifdef USE_CVI_TPU
        auto tpu_result = inference::TpuScheduler::instance().submit_float(
            config.session, input_data.data(), input_data.size(), input_shape);
        if (!tpu_result.success) {
            throw std::runtime_error("TPU inference failed: " + tpu_result.error);
        }
        result.outputs = std::move(tpu_result.outputs);
        result.output_shapes = std::move(tpu_result.output_shapes);
#else
        config.session->run_all(
            input_data.data(),
            input_shape,
            &result.outputs,
            &result.output_shapes);
#endif
        auto t_infer_end = std::chrono::steady_clock::now();
        result.infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        copy_run_stats(config.session->last_run_stats(),
                       &result.tpu_input_ms,
                       &result.tpu_forward_ms,
                       &result.tpu_output_ms);
    }
#else
    (void)frame;
    (void)roi;
    (void)config;
#endif

    return result;
}

}  // namespace node
