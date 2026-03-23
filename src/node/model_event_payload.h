#pragma once

#include <cstdint>

#include <nlohmann/json.hpp>

namespace node {

nlohmann::json build_inference_event_data(nlohmann::json result);

}  // namespace node
