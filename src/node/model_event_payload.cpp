#include "model_event_payload.h"

namespace node {

nlohmann::json build_inference_event_data(nlohmann::json result) {
    nlohmann::json event_data;
    if (result.is_object()) {
        event_data = std::move(result);
    } else if (result.is_array()) {
        event_data = nlohmann::json::object({{"boxes", std::move(result)}});
    } else {
        event_data = nlohmann::json::object({{"value", std::move(result)}});
    }
    return event_data;
}

}  // namespace node
