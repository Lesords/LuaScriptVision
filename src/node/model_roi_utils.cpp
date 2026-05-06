#include "model_roi_utils.h"

#include <algorithm>
#include <cmath>

namespace node {
namespace {

bool parse_keypoint(const nlohmann::json& item, FaceKeypoint* out) {
    if (!out) {
        return false;
    }

    FaceKeypoint keypoint;
    if (item.is_object()) {
        if (!item.contains("x") || !item.contains("y")) {
            return false;
        }
        keypoint.x = item.value("x", 0.0f);
        keypoint.y = item.value("y", 0.0f);
    } else if (item.is_array() && item.size() >= 2) {
        keypoint.x = item[0].get<float>();
        keypoint.y = item[1].get<float>();
    } else {
        return false;
    }

    if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y)) {
        return false;
    }

    *out = keypoint;
    return true;
}

void clamp_keypoint_to_bounds(int frame_width, int frame_height, FaceKeypoint* keypoint) {
    if (!keypoint || frame_width <= 0 || frame_height <= 0) {
        return;
    }

    keypoint->x = std::max(0.0f, std::min(keypoint->x, static_cast<float>(frame_width - 1)));
    keypoint->y = std::max(0.0f, std::min(keypoint->y, static_cast<float>(frame_height - 1)));
}

}  // namespace

bool parse_roi(const nlohmann::json& item, Roi* roi_out) {
    if (!roi_out) {
        return false;
    }

    Roi roi;
    if (item.is_object()) {
        if (item.contains("x") && item.contains("y") &&
            item.contains("w") && item.contains("h")) {
            roi.x = item.value("x", 0);
            roi.y = item.value("y", 0);
            roi.w = item.value("w", 0);
            roi.h = item.value("h", 0);
        } else if (item.contains("x1") && item.contains("y1") &&
                   item.contains("x2") && item.contains("y2")) {
            int x1 = item.value("x1", 0);
            int y1 = item.value("y1", 0);
            int x2 = item.value("x2", 0);
            int y2 = item.value("y2", 0);
            roi.x = x1;
            roi.y = y1;
            roi.w = x2 - x1;
            roi.h = y2 - y1;
        } else {
            return false;
        }
    } else if (item.is_array() && item.size() >= 4) {
        roi.x = item[0].get<int>();
        roi.y = item[1].get<int>();
        roi.w = item[2].get<int>();
        roi.h = item[3].get<int>();
    } else {
        return false;
    }

    if (roi.w <= 0 || roi.h <= 0) {
        return false;
    }
    *roi_out = roi;
    return true;
}

bool clamp_roi_to_bounds(int frame_width, int frame_height, Roi* roi) {
    if (!roi || frame_width <= 0 || frame_height <= 0) {
        return false;
    }

    roi->x = std::max(0, std::min(roi->x, frame_width - 1));
    roi->y = std::max(0, std::min(roi->y, frame_height - 1));
    roi->w = std::min(roi->w, frame_width - roi->x);
    roi->h = std::min(roi->h, frame_height - roi->y);

    return roi->w > 0 && roi->h > 0;
}

bool parse_selected_roi(const nlohmann::json& item, SelectedRoi* roi_out) {
    if (!roi_out) {
        return false;
    }

    SelectedRoi selected;
    if (!parse_roi(item, &selected.roi)) {
        return false;
    }

    selected.source = item;
    if (item.is_object() && item.contains("keypoints") &&
        item["keypoints"].is_array() && item["keypoints"].size() == selected.keypoints.size()) {
        bool parsed_all = true;
        for (size_t i = 0; i < selected.keypoints.size(); ++i) {
            if (!parse_keypoint(item["keypoints"][i], &selected.keypoints[i])) {
                parsed_all = false;
                break;
            }
        }
        selected.has_keypoints = parsed_all;
    }

    *roi_out = std::move(selected);
    return true;
}

bool clamp_selected_roi_to_bounds(int frame_width, int frame_height, SelectedRoi* roi) {
    if (!roi) {
        return false;
    }
    if (!clamp_roi_to_bounds(frame_width, frame_height, &roi->roi)) {
        return false;
    }
    if (roi->has_keypoints) {
        for (auto& keypoint : roi->keypoints) {
            clamp_keypoint_to_bounds(frame_width, frame_height, &keypoint);
        }
    }
    return true;
}

}  // namespace node
