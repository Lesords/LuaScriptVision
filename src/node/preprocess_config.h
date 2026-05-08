#pragma once

#include <array>
#include <string>

#include "LuaIntf.h"

namespace node {

// Preprocess configuration parsed from Lua script
struct PreprocessConfig {
    std::string type = "letterbox";     // "letterbox", "resize", "face_align", "none"
    int input_width = 640;
    int input_height = 640;
    int stride = 32;
    int fill_value = 114;
    bool center = true;
    bool normalize = true;
    float scale = 1.0f / 255.0f;
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> std = {1.0f, 1.0f, 1.0f};
    std::string format = "chw";         // "chw" or "hwc"
    std::string dtype = "float32";      // "float32" or "int8"

    // Debug: save preprocessed ROI crop to disk
    bool save_crop = false;
    std::string save_path = "/tmp";

    static PreprocessConfig fromLuaRef(LuaIntf::LuaRef& ref) {
        PreprocessConfig cfg;
        if (!ref.isTable()) {
            return cfg;
        }

        auto is_type = [](const LuaIntf::LuaRef& value, LuaIntf::LuaTypeID type) {
            return value.type() == type;
        };

        LuaIntf::LuaRef type_ref = ref["type"].value<LuaIntf::LuaRef>();
        if (is_type(type_ref, LuaIntf::LuaTypeID::STRING)) {
            cfg.type = type_ref.toValue<std::string>();
        }

        LuaIntf::LuaRef input_size = ref["input_size"].value<LuaIntf::LuaRef>();
        if (input_size.isTable()) {
            LuaIntf::LuaRef width_ref = input_size[1].value<LuaIntf::LuaRef>();
            if (is_type(width_ref, LuaIntf::LuaTypeID::NUMBER)) {
                cfg.input_width = width_ref.toValue<int>();
            }
            LuaIntf::LuaRef height_ref = input_size[2].value<LuaIntf::LuaRef>();
            if (is_type(height_ref, LuaIntf::LuaTypeID::NUMBER)) {
                cfg.input_height = height_ref.toValue<int>();
            }
        }

        LuaIntf::LuaRef stride_ref = ref["stride"].value<LuaIntf::LuaRef>();
        if (is_type(stride_ref, LuaIntf::LuaTypeID::NUMBER)) {
            cfg.stride = stride_ref.toValue<int>();
        }
        LuaIntf::LuaRef fill_ref = ref["fill_value"].value<LuaIntf::LuaRef>();
        if (is_type(fill_ref, LuaIntf::LuaTypeID::NUMBER)) {
            cfg.fill_value = fill_ref.toValue<int>();
        }
        LuaIntf::LuaRef center_ref = ref["center"].value<LuaIntf::LuaRef>();
        if (is_type(center_ref, LuaIntf::LuaTypeID::BOOLEAN)) {
            cfg.center = center_ref.toValue<bool>();
        }
        LuaIntf::LuaRef normalize_ref = ref["normalize"].value<LuaIntf::LuaRef>();
        if (is_type(normalize_ref, LuaIntf::LuaTypeID::BOOLEAN)) {
            cfg.normalize = normalize_ref.toValue<bool>();
        }
        LuaIntf::LuaRef scale_ref = ref["scale"].value<LuaIntf::LuaRef>();
        if (is_type(scale_ref, LuaIntf::LuaTypeID::NUMBER)) {
            cfg.scale = scale_ref.toValue<float>();
        }

        LuaIntf::LuaRef mean_ref = ref["mean"].value<LuaIntf::LuaRef>();
        if (mean_ref.isTable()) {
            for (int i = 0; i < 3; ++i) {
                LuaIntf::LuaRef value_ref = mean_ref[i + 1].value<LuaIntf::LuaRef>();
                if (is_type(value_ref, LuaIntf::LuaTypeID::NUMBER)) {
                    cfg.mean[i] = value_ref.toValue<float>();
                }
            }
        }

        LuaIntf::LuaRef std_ref = ref["std"].value<LuaIntf::LuaRef>();
        if (std_ref.isTable()) {
            for (int i = 0; i < 3; ++i) {
                LuaIntf::LuaRef value_ref = std_ref[i + 1].value<LuaIntf::LuaRef>();
                if (is_type(value_ref, LuaIntf::LuaTypeID::NUMBER)) {
                    cfg.std[i] = value_ref.toValue<float>();
                }
            }
        }

        LuaIntf::LuaRef format_ref = ref["format"].value<LuaIntf::LuaRef>();
        if (is_type(format_ref, LuaIntf::LuaTypeID::STRING)) {
            cfg.format = format_ref.toValue<std::string>();
        }
        LuaIntf::LuaRef dtype_ref = ref["dtype"].value<LuaIntf::LuaRef>();
        if (is_type(dtype_ref, LuaIntf::LuaTypeID::STRING)) {
            cfg.dtype = dtype_ref.toValue<std::string>();
        }

        LuaIntf::LuaRef save_crop_ref = ref["save_crop"].value<LuaIntf::LuaRef>();
        if (is_type(save_crop_ref, LuaIntf::LuaTypeID::BOOLEAN)) {
            cfg.save_crop = save_crop_ref.toValue<bool>();
        }
        LuaIntf::LuaRef save_path_ref = ref["save_path"].value<LuaIntf::LuaRef>();
        if (is_type(save_path_ref, LuaIntf::LuaTypeID::STRING)) {
            cfg.save_path = save_path_ref.toValue<std::string>();
        }

        return cfg;
    }
};

} // namespace node
