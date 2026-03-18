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

nlohmann::json build_roi_meta(const Roi& roi,
                              const nlohmann::json& upstream,
                              float threshold,
                              const PreprocessMeta& preprocess_meta) {
    nlohmann::json meta = {
        {"roi", {{"x", roi.x}, {"y", roi.y}, {"w", roi.w}, {"h", roi.h}}},
        {"upstream", upstream},
        {"threshold", threshold}
    };
    fill_meta_json(preprocess_meta, &meta);
    return meta;
}

}  // namespace node
