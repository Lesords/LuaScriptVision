#pragma once

#include <nlohmann/json.hpp>

namespace node {

struct Roi {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

bool parse_roi(const nlohmann::json& item, Roi* roi_out);
bool clamp_roi_to_bounds(int frame_width, int frame_height, Roi* roi);

}  // namespace node
