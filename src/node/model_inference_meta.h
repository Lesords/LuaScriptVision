#pragma once

#include "model_preprocess_utils.h"
#include "model_roi_utils.h"

#include <nlohmann/json.hpp>

namespace node {

nlohmann::json build_full_frame_meta(const nlohmann::json& upstream,
                                     float threshold,
                                     int frame_width,
                                     int frame_height,
                                     int output_count,
                                     const PreprocessMeta& preprocess_meta);

nlohmann::json build_roi_meta(const Roi& roi,
                              const nlohmann::json& upstream,
                              float threshold,
                              const PreprocessMeta& preprocess_meta);

}  // namespace node
