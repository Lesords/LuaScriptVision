#pragma once

#include "preprocess_config.h"

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace node {

struct PreprocessMeta {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int ori_w = 0;
    int ori_h = 0;
    int input_w = 0;
    int input_h = 0;
};

PreprocessMeta compute_letterbox_meta(int ori_w, int ori_h,
                                      int target_w, int target_h,
                                      bool center);

void fill_meta_json(const PreprocessMeta& meta, nlohmann::json* out);

void build_float_input(const cv::Mat& mat,
                       const PreprocessConfig& cfg,
                       std::vector<float>* data,
                       std::vector<int64_t>* shape);

}  // namespace node
