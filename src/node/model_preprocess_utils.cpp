#include "model_preprocess_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace node {
namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

}  // namespace

PreprocessMeta compute_letterbox_meta(int ori_w, int ori_h,
                                      int target_w, int target_h,
                                      bool center) {
    PreprocessMeta meta;
    meta.ori_w = ori_w;
    meta.ori_h = ori_h;
    meta.input_w = target_w;
    meta.input_h = target_h;

    float r = std::min(static_cast<float>(target_w) / ori_w,
                       static_cast<float>(target_h) / ori_h);
    int new_w = static_cast<int>(std::floor(ori_w * r));
    int new_h = static_cast<int>(std::floor(ori_h * r));

    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;

    int left = center ? pad_w / 2 : 0;
    int top = center ? pad_h / 2 : 0;

    meta.scale = r;
    meta.pad_x = left;
    meta.pad_y = top;
    return meta;
}

void fill_meta_json(const PreprocessMeta& meta, nlohmann::json* out) {
    if (!out) {
        return;
    }
    (*out)["scale"] = meta.scale;
    (*out)["pad_x"] = meta.pad_x;
    (*out)["pad_y"] = meta.pad_y;
    (*out)["ori_w"] = meta.ori_w;
    (*out)["ori_h"] = meta.ori_h;
    (*out)["input_w"] = meta.input_w;
    (*out)["input_h"] = meta.input_h;
}

void build_float_input(const cv::Mat& mat,
                       const PreprocessConfig& cfg,
                       std::vector<float>* data,
                       std::vector<int64_t>* shape) {
    if (!data || !shape) {
        throw std::invalid_argument("build_float_input - output buffers are null");
    }
    if (mat.empty()) {
        throw std::invalid_argument("build_float_input - input mat is empty");
    }

    int channels = mat.channels();
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("build_float_input - unsupported channel count");
    }

    float scale_val = cfg.normalize ? cfg.scale : 1.0f;
    std::array<float, 3> mean = cfg.normalize ? cfg.mean : std::array<float, 3>{0.0f, 0.0f, 0.0f};
    std::array<float, 3> stddev = cfg.normalize ? cfg.std : std::array<float, 3>{1.0f, 1.0f, 1.0f};

    std::array<float, 3> scale_factor{};
    std::array<float, 3> offset{};
    for (int c = 0; c < channels; ++c) {
        if (stddev[c] == 0.0f) {
            throw std::invalid_argument("build_float_input - stddev contains zero");
        }
        scale_factor[c] = scale_val / stddev[c];
        offset[c] = mean[c] / stddev[c];
    }

    const int height = mat.rows;
    const int width = mat.cols;
    const size_t plane_size = static_cast<size_t>(height) * width;

    std::string fmt = to_lower(cfg.format);
    if (fmt == "hwc") {
        data->assign(static_cast<size_t>(height) * width * channels, 0.0f);
        for (int y = 0; y < height; ++y) {
            const uint8_t* src = mat.ptr<uint8_t>(y);
            for (int x = 0; x < width; ++x) {
                size_t idx = (static_cast<size_t>(y) * width + x) * channels;
                if (channels == 1) {
                    data->at(idx) = src[x] * scale_factor[0] - offset[0];
                } else {
                    data->at(idx) = src[0] * scale_factor[0] - offset[0];
                    data->at(idx + 1) = src[1] * scale_factor[1] - offset[1];
                    data->at(idx + 2) = src[2] * scale_factor[2] - offset[2];
                    src += 3;
                }
            }
        }
        *shape = {1, height, width, channels};
        return;
    }

    data->assign(static_cast<size_t>(channels) * plane_size, 0.0f);
    if (channels == 1) {
        float* dst = data->data();
        for (int y = 0; y < height; ++y) {
            const uint8_t* src = mat.ptr<uint8_t>(y);
            size_t row_offset = static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                dst[row_offset + x] = src[x] * scale_factor[0] - offset[0];
            }
        }
    } else {
        float* dst0 = data->data();
        float* dst1 = data->data() + plane_size;
        float* dst2 = data->data() + plane_size * 2;

        for (int y = 0; y < height; ++y) {
            const uint8_t* src = mat.ptr<uint8_t>(y);
            size_t row_offset = static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                dst0[row_offset + x] = src[0] * scale_factor[0] - offset[0];
                dst1[row_offset + x] = src[1] * scale_factor[1] - offset[1];
                dst2[row_offset + x] = src[2] * scale_factor[2] - offset[2];
                src += 3;
            }
        }
    }
    *shape = {1, channels, height, width};
}

}  // namespace node
