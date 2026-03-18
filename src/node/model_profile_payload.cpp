#include "model_profile_payload.h"

namespace node {

nlohmann::json build_full_frame_profile_payload(const InferenceTimings& timings,
                                                const PreprocessMeta& preprocess_meta,
                                                double total_ms) {
    nlohmann::json profile = {
        {"mode", "full_frame"},
        {"preprocess_ms", timings.preprocess_ms},
        {"preprocess_vpss_ms", timings.vpss_ms},
        {"preprocess_cpu_ms", timings.cpu_pre_ms},
        {"build_input_ms", timings.build_input_ms},
        {"infer_ms", timings.infer_ms},
        {"postprocess_ms", timings.postprocess_ms},
        {"tpu_input_ms", timings.tpu_input_ms},
        {"tpu_forward_ms", timings.tpu_forward_ms},
        {"tpu_output_ms", timings.tpu_output_ms},
        {"preprocess_path", timings.preprocess_path},
        {"vpss_attempted", timings.vpss_attempted},
        {"use_vb", timings.use_vb},
        {"zero_copy", timings.use_vb},
        {"input_w", preprocess_meta.input_w},
        {"input_h", preprocess_meta.input_h},
        {"total_ms", total_ms}
    };
    return profile;
}

nlohmann::json build_cropped_roi_profile_payload(const RoiBatchMetrics& metrics,
                                                 double total_ms) {
    nlohmann::json profile = {
        {"mode", "cropped_roi"},
        {"roi_count", metrics.roi_count},
        {"preprocess_ms", metrics.preprocess_total_ms},
        {"preprocess_vpss_ms", metrics.vpss_total_ms},
        {"preprocess_cpu_ms", metrics.cpu_pre_total_ms},
        {"build_input_ms", metrics.build_input_total_ms},
        {"infer_ms", metrics.infer_total_ms},
        {"postprocess_ms", metrics.postprocess_total_ms},
        {"tpu_input_ms", metrics.tpu_input_total_ms},
        {"tpu_forward_ms", metrics.tpu_forward_total_ms},
        {"tpu_output_ms", metrics.tpu_output_total_ms},
        {"use_vb_count", metrics.use_vb_count},
        {"vpss_attempted_count", metrics.vpss_attempted_count},
        {"total_ms", total_ms},
        {"avg_preprocess_ms", metrics.preprocess_total_ms / metrics.roi_count},
        {"avg_infer_ms", metrics.infer_total_ms / metrics.roi_count},
        {"avg_postprocess_ms", metrics.postprocess_total_ms / metrics.roi_count}
    };
    return profile;
}

}  // namespace node
