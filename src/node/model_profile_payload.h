#pragma once

#include "model_preprocess_utils.h"

#include <nlohmann/json.hpp>
#include <string>

namespace node {

struct InferenceTimings {
    double preprocess_ms = 0.0;
    double vpss_ms = 0.0;
    double cpu_pre_ms = 0.0;
    double build_input_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double tpu_input_ms = 0.0;
    double tpu_forward_ms = 0.0;
    double tpu_output_ms = 0.0;
    bool vpss_attempted = false;
    bool use_vb = false;
    std::string preprocess_path = "cpu";
};

struct RoiBatchMetrics {
    double preprocess_total_ms = 0.0;
    double vpss_total_ms = 0.0;
    double cpu_pre_total_ms = 0.0;
    double build_input_total_ms = 0.0;
    double infer_total_ms = 0.0;
    double postprocess_total_ms = 0.0;
    double tpu_input_total_ms = 0.0;
    double tpu_forward_total_ms = 0.0;
    double tpu_output_total_ms = 0.0;
    int roi_count = 0;
    int use_vb_count = 0;
    int vpss_attempted_count = 0;
};

nlohmann::json build_full_frame_profile_payload(const InferenceTimings& timings,
                                                const PreprocessMeta& preprocess_meta,
                                                double total_ms);

nlohmann::json build_cropped_roi_profile_payload(const RoiBatchMetrics& metrics,
                                                 double total_ms);

}  // namespace node
