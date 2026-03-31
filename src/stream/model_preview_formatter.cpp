#include "model_preview_formatter.h"

#include <opencv2/opencv.hpp>

namespace node {
namespace {

static const char* kBase64Chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const unsigned char* data, size_t len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (len--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) {
                ret += kBase64Chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i > 0) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++) {
            ret += kBase64Chars[char_array_4[j]];
        }

        while (i++ < 3) {
            ret += '=';
        }
    }

    return ret;
}

void populate_preview_boxes(const nlohmann::json& event_data, nlohmann::json* preview_data) {
    if (!preview_data || !event_data.contains("boxes") || !event_data["boxes"].is_array()) {
        return;
    }

    nlohmann::json boxes_arr = nlohmann::json::array();
    nlohmann::json labels_arr = nlohmann::json::array();
    for (const auto& box : event_data["boxes"]) {
        if (!box.is_object()) {
            continue;
        }
        double x = box.value("x", 0.0);
        double y = box.value("y", 0.0);
        double w = box.value("w", 0.0);
        double h = box.value("h", 0.0);
        double score = box.value("score", 0.0);
        int cls = box.value("class_id", 0);
        boxes_arr.push_back({x, y, w, h, score, cls});
        labels_arr.push_back(box.value("label", ""));
    }

    (*preview_data)["boxes"] = std::move(boxes_arr);
    (*preview_data)["labels"] = std::move(labels_arr);
}

std::string encode_preview_image(const lua_cv::Frame& frame,
                                 const ModelPreviewFormatConfig& config) {
    cv::Mat mat = frame.to_mat_copy();
    if (mat.empty()) {
        return "";
    }

    if (config.preview_width > 0 && config.preview_height > 0 &&
        (mat.cols != config.preview_width || mat.rows != config.preview_height)) {
        cv::Mat resized;
        cv::resize(mat, resized, cv::Size(config.preview_width, config.preview_height), 0, 0, cv::INTER_LINEAR);
        mat = std::move(resized);
    }

    std::vector<uchar> jpeg_buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config.jpeg_quality};
    if (!cv::imencode(".jpg", mat, jpeg_buf, params)) {
        return "";
    }
    return base64_encode(jpeg_buf.data(), jpeg_buf.size());
}

}  // namespace

// Public: encode raw bytes to base64 (for hardware-encoded JPEG preview)
std::string base64_encode_stream(const uint8_t* data, size_t len) {
    return base64_encode(data, len);
}

// Public: build preview JSON from pre-encoded base64 JPEG + inference boxes
nlohmann::json build_preview_json(const nlohmann::json& event_data,
                                  const std::string& base64_jpeg,
                                  uint32_t width, uint32_t height) {
    nlohmann::json preview_data = nlohmann::json::object();
    populate_preview_boxes(event_data, &preview_data);
    preview_data["resolution"] = {width, height};
    preview_data["image"] = base64_jpeg;
    return preview_data;
}

nlohmann::json build_model_preview_message(
    const nlohmann::json& event_data,
    const lua_cv::Frame& frame,
    const ModelPreviewFormatConfig& config,
    std::chrono::steady_clock::time_point* last_preview_time,
    int* preview_interval_ms) {
    nlohmann::json preview_data = nlohmann::json::object();
    populate_preview_boxes(event_data, &preview_data);

    int frame_width = config.preview_width > 0 ? config.preview_width : event_data.value("frame_width", 0);
    int frame_height = config.preview_height > 0 ? config.preview_height : event_data.value("frame_height", 0);
    preview_data["resolution"] = {frame_width, frame_height};

    try {
        if (preview_interval_ms) {
            *preview_interval_ms = 1000 / config.preview_fps;
        }

        auto now_tp = std::chrono::steady_clock::now();
        auto elapsed = last_preview_time
            ? std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - *last_preview_time).count()
            : 0;

        if (!last_preview_time || elapsed >= (preview_interval_ms ? *preview_interval_ms : 0)) {
            if (last_preview_time) {
                *last_preview_time = now_tp;
            }
            preview_data["image"] = encode_preview_image(frame, config);
        } else {
            preview_data["image"] = "";
        }
    } catch (...) {
        preview_data["image"] = "";
    }

    return {{"data", std::move(preview_data)}};
}

#ifdef USE_CVI_MPI
nlohmann::json build_model_preview_message(
    const nlohmann::json& event_data,
    const lua_cv::VencEncoder::EncodedStream& stream,
    const ModelPreviewFormatConfig& config,
    std::chrono::steady_clock::time_point* last_preview_time,
    int* preview_interval_ms) {
    nlohmann::json preview_data = nlohmann::json::object();
    populate_preview_boxes(event_data, &preview_data);

    int frame_width = config.preview_width > 0 ? config.preview_width : event_data.value("frame_width", 0);
    int frame_height = config.preview_height > 0 ? config.preview_height : event_data.value("frame_height", 0);
    preview_data["resolution"] = {frame_width, frame_height};

    try {
        if (preview_interval_ms) {
            *preview_interval_ms = 1000 / config.preview_fps;
        }

        auto now_tp = std::chrono::steady_clock::now();
        auto elapsed = last_preview_time
            ? std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - *last_preview_time).count()
            : 0;

        if (!last_preview_time || elapsed >= (preview_interval_ms ? *preview_interval_ms : 0)) {
            if (last_preview_time) {
                *last_preview_time = now_tp;
            }
            if (!stream.data.empty()) {
                preview_data["image"] = base64_encode(stream.data.data(), stream.data.size());
            } else {
                preview_data["image"] = "";
            }
        } else {
            preview_data["image"] = "";
        }
    } catch (...) {
        preview_data["image"] = "";
    }

    return {{"data", std::move(preview_data)}};
}
#endif

}  // namespace node
