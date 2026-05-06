#pragma once

#include <array>

#include <nlohmann/json.hpp>

namespace node {

struct Roi {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

struct FaceKeypoint {
    float x = 0.0f;
    float y = 0.0f;
};

struct SelectedRoi {
    Roi roi;
    std::array<FaceKeypoint, 5> keypoints{};
    bool has_keypoints = false;
    nlohmann::json source = nlohmann::json::object();
};

bool parse_roi(const nlohmann::json& item, Roi* roi_out);
bool clamp_roi_to_bounds(int frame_width, int frame_height, Roi* roi);
bool parse_selected_roi(const nlohmann::json& item, SelectedRoi* roi_out);
bool clamp_selected_roi_to_bounds(int frame_width, int frame_height, SelectedRoi* roi);

}  // namespace node
