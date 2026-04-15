#pragma once

#include "modules/cv/frame.h"

#include <chrono>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#ifdef USE_CVI_MPI
#include "stream/venc_encoder.h"
#endif

namespace node {

struct ModelPreviewFormatConfig {
    int preview_width = 0;
    int preview_height = 0;
    int preview_fps = 15;
    int jpeg_quality = 75;
};

// Encode raw bytes to base64 string (used for JPEG preview payload)
std::string base64_encode_stream(const uint8_t* data, size_t len);

// Build preview JSON payload from pre-encoded base64 JPEG + inference boxes
// src_width/src_height: coordinate space of the box data (e.g. camera resolution)
// width/height: actual preview JPEG dimensions
nlohmann::json build_preview_json(const nlohmann::json& event_data,
                                  const std::string& base64_jpeg,
                                  uint32_t width, uint32_t height,
                                  uint32_t src_width, uint32_t src_height);

nlohmann::json build_model_preview_message(
    const nlohmann::json& event_data,
    const lua_cv::Frame& frame,
    const ModelPreviewFormatConfig& config,
    std::chrono::steady_clock::time_point* last_preview_time,
    int* preview_interval_ms);

#ifdef USE_CVI_MPI
nlohmann::json build_model_preview_message(
    const nlohmann::json& event_data,
    const lua_cv::VencEncoder::EncodedStream& stream,
    const ModelPreviewFormatConfig& config,
    std::chrono::steady_clock::time_point* last_preview_time,
    int* preview_interval_ms);
#endif

}  // namespace node
