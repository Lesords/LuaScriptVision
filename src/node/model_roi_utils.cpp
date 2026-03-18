#include "model_roi_utils.h"

#include <algorithm>

namespace node {

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

}  // namespace node
