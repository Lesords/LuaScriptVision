#pragma once

#include "model_preprocess_utils.h"
#include "model_profile_payload.h"
#include "model_roi_utils.h"
#include "preprocess_config.h"

#include <optional>
#include <string>
#include <vector>

namespace inference { class CviSession; }
namespace lua_cv { class Frame; class CviVpssProcessor; }

namespace node {

struct ExecutionWarning {
    std::string message;
    std::string detail;
};

struct ModelExecutorConfig {
    inference::CviSession* session = nullptr;
    const PreprocessConfig* preprocess_config = nullptr;
    bool crop_size_explicit = false;
    int crop_width = 0;
    int crop_height = 0;
    // Persistent VPSS processor: reused across frames to avoid per-frame
    // CVI_VPSS_CreateGrp/DestroyGrp which leaks ION work buffers in the driver.
    // Owned by ModelNode, lifetime matches inference session.
    lua_cv::CviVpssProcessor* vpss_processor = nullptr;
};

struct FullFrameExecutionResult {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> output_shapes;
    PreprocessMeta preprocess_meta;
    InferenceTimings timings;
    std::optional<ExecutionWarning> warning;
};

struct RoiExecutionResult {
    bool valid = true;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> output_shapes;
    PreprocessMeta preprocess_meta;
    double preprocess_ms = 0.0;
    double vpss_ms = 0.0;
    double cpu_pre_ms = 0.0;
    double build_input_ms = 0.0;
    double infer_ms = 0.0;
    double tpu_input_ms = 0.0;
    double tpu_forward_ms = 0.0;
    double tpu_output_ms = 0.0;
    bool vpss_attempted = false;
    bool use_vb = false;
    std::optional<ExecutionWarning> warning;
};

FullFrameExecutionResult execute_full_frame_inference(const lua_cv::Frame& frame,
                                                      const ModelExecutorConfig& config);

RoiExecutionResult execute_roi_inference(const lua_cv::Frame& frame,
                                         const Roi& roi,
                                         const ModelExecutorConfig& config);

}  // namespace node
