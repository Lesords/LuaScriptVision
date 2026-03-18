#include "model_event_payload.h"

namespace node {

nlohmann::json build_inference_event_data(nlohmann::json result,
                                          uint64_t frame_id,
                                          int frame_width,
                                          int frame_height) {
    nlohmann::json event_data;
    if (result.is_object()) {
        event_data = std::move(result);
    } else if (result.is_array()) {
        event_data = nlohmann::json::object({{"boxes", std::move(result)}});
    } else {
        event_data = nlohmann::json::object({{"value", std::move(result)}});
    }

    event_data["frame_id"] = frame_id;
    if (!event_data.contains("frame_width")) {
        event_data["frame_width"] = frame_width;
    }
    if (!event_data.contains("frame_height")) {
        event_data["frame_height"] = frame_height;
    }
    return event_data;
}

}  // namespace node
