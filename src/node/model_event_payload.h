#pragma once

#include <cstdint>

#include <nlohmann/json.hpp>

namespace node {

nlohmann::json build_inference_event_data(nlohmann::json result,
                                          uint64_t frame_id,
                                          int frame_width,
                                          int frame_height);

}  // namespace node
