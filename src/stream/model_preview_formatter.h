#pragma once

#include "modules/cv/frame.h"

#include <chrono>

#include <nlohmann/json.hpp>

namespace node {

struct ModelPreviewFormatConfig {
    int preview_width = 0;
    int preview_height = 0;
    int preview_fps = 15;
    int jpeg_quality = 75;
};

nlohmann::json build_model_preview_message(
    const nlohmann::json& event_data,
    const lua_cv::Frame& frame,
    const ModelPreviewFormatConfig& config,
    std::chrono::steady_clock::time_point* last_preview_time,
    int* preview_interval_ms);

}  // namespace node
