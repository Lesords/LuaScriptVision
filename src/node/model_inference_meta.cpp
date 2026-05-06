#include "model_inference_meta.h"

namespace node {

nlohmann::json build_full_frame_meta(const nlohmann::json& upstream,
                                     float threshold,
                                     int frame_width,
                                     int frame_height,
                                     int output_count,
                                     const PreprocessMeta& preprocess_meta) {
    nlohmann::json meta = {
        {"upstream", upstream},
        {"threshold", threshold},
        {"frame_width", frame_width},
        {"frame_height", frame_height},
        {"output_count", output_count}
    };
    fill_meta_json(preprocess_meta, &meta);
    return meta;
}

namespace {

nlohmann::json build_keypoints_json(const SelectedRoi& roi) {
    nlohmann::json keypoints = nlohmann::json::array();
    if (!roi.has_keypoints) {
        return keypoints;
    }
    for (const auto& keypoint : roi.keypoints) {
        keypoints.push_back({{"x", keypoint.x}, {"y", keypoint.y}});
    }
    return keypoints;
}

}  // namespace

nlohmann::json build_roi_meta(const SelectedRoi& roi,
                              const nlohmann::json& upstream,
                              float threshold,
                              const PreprocessMeta& preprocess_meta) {
    nlohmann::json source = roi.source;
    if (!source.is_object()) {
        source = nlohmann::json::object();
    }
    source["x"] = roi.roi.x;
    source["y"] = roi.roi.y;
    source["w"] = roi.roi.w;
    source["h"] = roi.roi.h;
    if (roi.has_keypoints) {
        source["keypoints"] = build_keypoints_json(roi);
    }

    nlohmann::json meta = {
        {"roi", {{"x", roi.roi.x}, {"y", roi.roi.y}, {"w", roi.roi.w}, {"h", roi.roi.h}}},
        {"source", std::move(source)},
        {"upstream", upstream},
        {"threshold", threshold}
    };
    fill_meta_json(preprocess_meta, &meta);
    return meta;
}

}  // namespace node
