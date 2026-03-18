#include "model_node.h"
#include "node_factory.h"
#include "node_server.h"
#include "camera_node.h"
#include "resource_estimator.h"
#include "stream/websocket_transport.h"
#include "modules/cv/cv_helpers.h"
#include "modules/cv/cv_types.h"
#include "tensor/tensor.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sys/stat.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include <opencv2/opencv.hpp>

#ifdef USE_CVI_TPU
#include "inference/cvi_session.h"
#endif

#ifdef USE_CVI_MPI
#include "modules/cv/cvi_vpss_processor.h"
#endif

// Forward declaration of module registration functions
namespace lua_cv { void register_module(lua_State* L); }
namespace lua_nn { void register_module(lua_State* L); }
namespace lua_utils { void register_module(lua_State* L); }

namespace node {

namespace {
double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct PreprocessMeta {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int ori_w = 0;
    int ori_h = 0;
    int input_w = 0;
    int input_h = 0;
};

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

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

// Base64 encoding for WebSocket image transmission
static const char* kBase64Chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_encode(const unsigned char* data, size_t len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (len--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) {
                ret += kBase64Chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i > 0) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++) {
            ret += kBase64Chars[char_array_4[j]];
        }

        while (i++ < 3) {
            ret += '=';
        }
    }

    return ret;
}

// Encode frame to base64 JPEG for WebSocket transmission
static std::string encode_frame_to_base64_jpeg(const lua_cv::Frame& frame, int quality = 80) {
    try {
        cv::Mat mat = frame.to_mat_copy();
        if (mat.empty()) {
            return "";
        }

        // Convert to BGR if needed
        if (mat.channels() == 1) {
            cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
        } else if (mat.channels() == 4) {
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
        }

        std::vector<uint8_t> buffer;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
        cv::imencode(".jpg", mat, buffer, params);

        return base64_encode(buffer.data(), buffer.size());
    } catch (const std::exception& e) {
        std::cerr << "[ModelNode] Failed to encode frame: " << e.what() << std::endl;
        return "";
    }
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

void push_json(lua_State* L, const nlohmann::json& value) {
    if (value.is_null()) {
        lua_pushnil(L);
    } else if (value.is_boolean()) {
        lua_pushboolean(L, value.get<bool>());
    } else if (value.is_number_integer()) {
        lua_pushinteger(L, static_cast<lua_Integer>(value.get<int64_t>()));
    } else if (value.is_number_unsigned()) {
        lua_pushinteger(L, static_cast<lua_Integer>(value.get<uint64_t>()));
    } else if (value.is_number_float()) {
        lua_pushnumber(L, static_cast<lua_Number>(value.get<double>()));
    } else if (value.is_string()) {
        lua_pushstring(L, value.get<std::string>().c_str());
    } else if (value.is_array()) {
        lua_newtable(L);
        int index = 1;
        for (const auto& item : value) {
            push_json(L, item);
            lua_rawseti(L, -2, index++);
        }
    } else if (value.is_object()) {
        lua_newtable(L);
        for (auto it = value.begin(); it != value.end(); ++it) {
            lua_pushstring(L, it.key().c_str());
            push_json(L, it.value());
            lua_settable(L, -3);
        }
    } else {
        lua_pushnil(L);
    }
}

LuaIntf::LuaRef json_to_luaref(lua_State* L, const nlohmann::json& value) {
    push_json(L, value);
    return LuaIntf::LuaRef::popFromStack(L);
}

struct Roi {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

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
}  // namespace

// Register ModelNode type
REGISTER_NODE("model", ModelNode);

ModelNode::ModelNode(const std::string& id, const std::string& type)
    : DataNode(id, type, 1) {}

ModelNode::~ModelNode() {
    onDestroy();
}

int ModelNode::onCreate(const nlohmann::json& config) {
    // 1. Parse configuration
    // Support both 'model' (internal) and 'uri' (Node-RED sscma-node) fields
    if (config.contains("model")) {
        config_.model_path = config.at("model");
    } else if (config.contains("uri")) {
        config_.model_path = config.at("uri");
    } else {
        // Default model path for Node-RED compatibility
        config_.model_path = "/usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel";
    }

#ifdef USE_CVI_TPU
    {
        struct stat st {};
        if (stat(config_.model_path.c_str(), &st) != 0) {
            last_error_ = "Model file not found: " + config_.model_path;
            return MA_ENOENT;
        }
    }
#endif

    // Support both 'script' (internal) and use default for Node-RED compatibility
    if (config.contains("script")) {
        config_.script_path = config.at("script");
    } else {
        // Default script path for Node-RED compatibility
        config_.script_path = "/userdata/scripts/yolo11_tensor_detector.lua";
    }

    // Verify script file exists
    {
        struct stat st {};
        if (stat(config_.script_path.c_str(), &st) != 0) {
            last_error_ = "Script file not found: " + config_.script_path;
            return MA_ENOENT;
        }
    }

    // Support both 'threshold' (internal) and 'tscore' (Node-RED sscma-node) fields
    if (config.contains("threshold")) {
        config_.conf_threshold = config["threshold"].get<float>();
    } else if (config.contains("tscore")) {
        config_.conf_threshold = config["tscore"].get<float>();
    }
    if (config.contains("input_mode")) {
        std::string mode = config["input_mode"];
        config_.input_mode = (mode == "cropped_roi") ? CROPPED_ROI : FULL_FRAME;
    }
    if (config.contains("crop_size") && config["crop_size"].is_array()) {
        config_.crop_width = config["crop_size"][0].get<int>();
        config_.crop_height = config["crop_size"][1].get<int>();
        config_.crop_size_explicit = true;
    }
    if (config.contains("timeout_ms")) {
        config_.infer_timeout_ms = config["timeout_ms"].get<int>();
    }
    if (config.contains("profile")) {
        config_.profile = config["profile"].get<bool>();
    } else {
        const char* env = std::getenv("MODEL_NODE_PROFILE");
        if (env && std::string(env) == "1") {
            config_.profile = true;
        }
    }
    if (config.contains("websocket")) {
        config_.websocket = config["websocket"].get<bool>();
    }
    if (config.contains("ws_port")) {
        config_.ws_port = config["ws_port"].get<int>();
    }
    if (config.contains("ws_path")) {
        config_.ws_path = config["ws_path"].get<std::string>();
    }
    if (config.contains("ws_max_clients")) {
        config_.ws_max_clients = config["ws_max_clients"].get<int>();
    }
    if (config.contains("output")) {
        config_.output = config["output"].get<bool>();
    }
    if (config.contains("debug")) {
        config_.debug = config["debug"].get<bool>();
    }

    if (config_.debug) {
        config_.output = true;
    }

    // Preview video stream configuration
    if (config.contains("preview_resolution") && config["preview_resolution"].is_string()) {
        config_.preview_resolution = config["preview_resolution"].get<std::string>();

        // 解析 "WIDTHxHEIGHT" 格式（如 "640x640"）
        std::string res = config_.preview_resolution;
        size_t sep_pos = res.find_first_of("xX");
        if (sep_pos != std::string::npos && sep_pos > 0 && sep_pos < res.length() - 1) {
            try {
                config_.preview_width = std::stoi(res.substr(0, sep_pos));
                config_.preview_height = std::stoi(res.substr(sep_pos + 1));

                // 验证分辨率
                if (config_.preview_width > 0 && config_.preview_height > 0) {
                    std::cout << "[ModelNode] Preview resolution: " << config_.preview_width
                              << "x" << config_.preview_height << std::endl;
                } else {
                    std::cerr << "[ModelNode] Invalid preview_resolution format, using original size" << std::endl;
                    config_.preview_width = 0;
                    config_.preview_height = 0;
                }
            } catch (const std::exception& e) {
                std::cerr << "[ModelNode] Failed to parse preview_resolution: " << e.what() << std::endl;
                config_.preview_width = 0;
                config_.preview_height = 0;
            }
        } else {
            std::cerr << "[ModelNode] Invalid preview_resolution format (expected WIDTHxHEIGHT), using original size" << std::endl;
            config_.preview_width = 0;
            config_.preview_height = 0;
        }
    }

    if (config.contains("preview_fps")) {
        config_.preview_fps = config["preview_fps"].get<int>();
        if (config_.preview_fps <= 0 || config_.preview_fps > 30) {
            std::cerr << "[ModelNode] Invalid preview_fps, using default 15" << std::endl;
            config_.preview_fps = 15;
        }
    }

    if (config.contains("jpeg_quality")) {
        config_.jpeg_quality = config["jpeg_quality"].get<int>();
        if (config_.jpeg_quality < 1 || config_.jpeg_quality > 100) {
            std::cerr << "[ModelNode] Invalid jpeg_quality, using default 75" << std::endl;
            config_.jpeg_quality = 75;
        }
    }

    // 2. Set LUA_PATH from script directory if not already set
    const char* existing_lua_path = std::getenv("LUA_PATH");
    if (!existing_lua_path) {
        // Build LUA_PATH including both the script directory and its parent.
        // Scripts often use require("scripts.lib.foo") which resolves relative to
        // the parent of the scripts/ directory, so we must include parent_dir/?.lua.
        size_t last_slash = config_.script_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            std::string script_dir = config_.script_path.substr(0, last_slash);
            std::string parent_dir;
            size_t parent_slash = script_dir.find_last_of("/\\");
            if (parent_slash != std::string::npos) {
                parent_dir = script_dir.substr(0, parent_slash);
            }
            std::string lua_path;
            if (!parent_dir.empty()) {
                lua_path = parent_dir + "/?.lua;" + parent_dir + "/?/init.lua;";
            }
            lua_path += script_dir + "/?.lua;" + script_dir + "/?/init.lua;;";
            setenv("LUA_PATH", lua_path.c_str(), 1);
        }
    }

    // 3. Create independent Lua State
    L_ = luaL_newstate();
    if (!L_) {
        last_error_ = "Failed to create Lua state";
        return MA_ENOMEM;
    }
    luaL_openlibs(L_);

    // Register modules
    lua_cv::register_module(L_);
    lua_nn::register_module(L_);
    lua_utils::register_module(L_);

    // 3. Load Lua script
    if (luaL_dofile(L_, config_.script_path.c_str()) != LUA_OK) {
        std::string err = lua_tostring(L_, -1);
        last_error_ = "Lua script error: " + err;
        lua_close(L_);
        L_ = nullptr;
        return MA_EINVAL;
    }

    // 4. Extract Model table (script should return a table)
    LuaIntf::LuaRef model = LuaIntf::LuaRef::popFromStack(L_);
    auto cleanup_with_model = [&](int code) {
        model = LuaIntf::LuaRef();
        cleanupLuaRef();
        return code;
    };
    if (!model.isTable()) {
        last_error_ = "Script must return a table";
        return cleanup_with_model(MA_EINVAL);
    }

    postprocess_ = model["postprocess"];
    select_rois_ = model["select_rois"];
    preprocess_config_ref_ = model["preprocess_config"];

    if (!postprocess_.isFunction()) {
        last_error_ = "Missing postprocess function in script";
        return cleanup_with_model(MA_EINVAL);
    }

    // Parse preprocess config if available
    if (preprocess_config_ref_.isTable()) {
        preprocess_config_ = PreprocessConfig::fromLuaRef(preprocess_config_ref_);
        preprocess_config_explicit_ = true;
    }

    // 5. Load TPU model
#ifdef USE_CVI_TPU
    try {
        session_ = std::make_unique<inference::CviSession>(config_.model_path);

        // Validate model input matches preprocess config
        if (preprocess_config_ref_.isTable()) {
            auto model_shape = session_->get_input_shape(0);
            if (model_shape.size() >= 4) {
                int model_h = static_cast<int>(model_shape[2]);
                int model_w = static_cast<int>(model_shape[3]);
                if (preprocess_config_.input_width != model_w ||
                    preprocess_config_.input_height != model_h) {
                    event("warning", 0, {
                        {"message", "Input size mismatch"},
                        {"script", {preprocess_config_.input_width, preprocess_config_.input_height}},
                        {"model", {model_w, model_h}}
                    });
                }
            }
        }

        if (!preprocess_config_explicit_) {
            if (session_->supports_vb_input()) {
                auto spec = session_->get_vb_input_spec();
                preprocess_config_.input_width = static_cast<int>(spec.width);
                preprocess_config_.input_height = static_cast<int>(spec.height);
            } else {
                auto model_shape = session_->get_input_shape(0);
                if (model_shape.size() >= 4) {
                    preprocess_config_.input_height = static_cast<int>(model_shape[2]);
                    preprocess_config_.input_width = static_cast<int>(model_shape[3]);
                }
            }
        }
    } catch (const std::exception& e) {
        last_error_ = "Model load failed: " + std::string(e.what());
        return cleanup_with_model(MA_EIO);
    }
#else
    // CPU-only build: validate file exists
    FILE* f = fopen(config_.model_path.c_str(), "r");
    if (!f) {
        last_error_ = "Model file not found: " + config_.model_path;
        cleanupLuaRef();
        return MA_ENOENT;
    }
    fclose(f);
#endif

    // 6. Validate CROPPED_ROI mode constraints
    // CROPPED_ROI mode requires upstream ModelNode and select_rois in script
    if (config_.input_mode == CROPPED_ROI) {
        bool has_model_upstream = false;
        for (const auto& [dep_id, dep] : dependencies_) {
            if (dep->type() == "model") {
                has_model_upstream = true;
                break;
            }
        }
        if (!has_model_upstream) {
            last_error_ = "CROPPED_ROI mode requires upstream ModelNode";
            return cleanup_with_model(MA_EINVAL);
        }
        if (!select_rois_.isFunction()) {
            last_error_ = "CROPPED_ROI mode requires select_rois(upstream) in script";
            return cleanup_with_model(MA_EINVAL);
        }
    }

    // 7. Find upstream CameraNode for timing feedback
    for (const auto& [dep_id, dep] : dependencies_) {
        if (dep->type() == "camera") {
            upstream_camera_ = static_cast<CameraNode*>(dep);
            break;
        }
    }

    return MA_OK;
}

int ModelNode::onStart() {
    if (running_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

#ifdef USE_CVI_TPU
    // Verify session was created in onCreate
    if (!session_) {
        last_error_ = "CviSession not initialized";
        return MA_EINVAL;
    }
#endif

    inbox_.clear();
    inbox_.reset();

    // Register resource usage
    ResourceRequirement usage;
    usage.vb_infer_blocks = 1;  // Each model uses ~1 VB block

    std::string camera_id;
    std::string upstream_model_id;
    for (const auto& [dep_id, dep] : dependencies_) {
        if (!dep) {
            continue;
        }
        if (dep->type() == "camera") {
            camera_id = dep_id;
        } else if (dep->type() == "model") {
            upstream_model_id = dep_id;
        }
    }

    ResourceEstimator::instance().on_node_started(
        id_, usage, camera_id, upstream_model_id);

    running_.store(true, std::memory_order_release);

    if (config_.websocket) {
        lua_cv::WebSocketTransport::Config ws_cfg;
        ws_cfg.port = config_.ws_port;
        ws_cfg.path = config_.ws_path;
        ws_cfg.max_clients = config_.ws_max_clients;
        ws_ = std::make_unique<lua_cv::WebSocketTransport>(ws_cfg);
        if (!ws_->start()) {
            last_error_ = "ModelNode WebSocket start failed";
            ws_.reset();
            running_.store(false, std::memory_order_release);
            ResourceEstimator::instance().on_node_stopped(id_);
            return MA_EIO;
        }
        event("websocket", MA_OK, {
            {"port", config_.ws_port},
            {"path", config_.ws_path},
            {"type", "json"}
        });
    }

    infer_thread_ = std::thread(&ModelNode::inferLoop, this);

    // Send enabled event to notify frontend of initial state
    event("enabled", MA_OK, infer_enabled_.load(std::memory_order_acquire));

    return MA_OK;
}

int ModelNode::onStop() {
    if (!running_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

    running_.store(false, std::memory_order_release);
    inbox_.interrupt();  // Wake up blocking fetch

    if (infer_thread_.joinable()) {
        infer_thread_.join();
    }

    if (ws_) {
        ws_->stop();
        ws_.reset();
    }

    inbox_.clear();

    // Unregister resource usage
    ResourceEstimator::instance().on_node_stopped(id_);

    // Send enabled event to notify frontend of stopped state
    event("enabled", MA_OK, false);

    return MA_OK;
}

int ModelNode::onDestroy() {
    onStop();

    session_.reset();
    cleanupLuaRef();

    return MA_OK;
}

int ModelNode::onControl(const std::string& action, const nlohmann::json& data) {
    if (action == "set_threshold") {
        if (!data.contains("value")) {
            return MA_EINVAL;
        }
        config_.conf_threshold = data.at("value").get<float>();
        return MA_OK;
    }

    if (action == "get_stats") {
        response("get_stats", MA_OK, {
            {"infer_count", infer_count_.load()},
            {"error_count", error_count_.load()},
            {"infer_ema_ms", infer_ema_ms_},
            {"ws_event_count", ws_event_count_.load()},
            {"ws_clients", ws_ ? ws_->client_count() : 0}
        });
        return MA_OK;
    }

    if (action == "reload_script") {
        if (running_.load(std::memory_order_acquire)) {
            return MA_EBUSY;
        }
        // Would need to re-run onCreate logic
        return MA_OK;
    }

    if (action == "enabled") {
        // Support both formats: {"value": bool} or direct boolean
        bool enabled;
        if (data.is_boolean()) {
            enabled = data.get<bool>();
        } else if (data.contains("value")) {
            enabled = data["value"].get<bool>();
        } else {
            enabled = true;  // Default to true
        }
        infer_enabled_.store(enabled, std::memory_order_release);
        event("enabled", MA_OK, enabled);  // Send direct boolean for frontend
        return MA_OK;
    }

    return MA_EINVAL;
}

void ModelNode::inferLoop() {
    while (running_.load(std::memory_order_acquire)) {
        PipelineContext* ctx;
        if (!inbox_.fetch(&ctx, 100)) {
            // Timeout or interrupted
            continue;
        }

        // When disabled, drain inbox without processing
        if (!infer_enabled_.load(std::memory_order_acquire)) {
            delete ctx;
            continue;
        }

        auto t_start = std::chrono::steady_clock::now();

        try {
            nlohmann::json result = runInference(ctx->frame->frame(), ctx->upstream_result);

            auto t_end = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            // Check timeout (soft timeout - just warn)
            if (elapsed_ms > config_.infer_timeout_ms) {
                event("warning", 0, {
                    {"message", "Inference timeout"},
                    {"elapsed_ms", elapsed_ms},
                    {"timeout_ms", config_.infer_timeout_ms}
                });
            }

            // Update statistics
            infer_count_.fetch_add(1, std::memory_order_relaxed);
            updateEwma(elapsed_ms);

            // Report timing to upstream camera
            if (upstream_camera_) {
                upstream_camera_->report_proc_time(id_, elapsed_ms);
            }

            nlohmann::json event_data;
            if (result.is_object()) {
                event_data = std::move(result);
            } else if (result.is_array()) {
                event_data = nlohmann::json::object({{"boxes", std::move(result)}});
            } else {
                event_data = nlohmann::json::object({{"value", std::move(result)}});
            }

            event_data["latency_ms"] = elapsed_ms;
            event_data["frame_id"] = ctx->frame_id;
            if (!event_data.contains("frame_width")) {
                event_data["frame_width"] = ctx->frame->frame().width();
            }
            if (!event_data.contains("frame_height")) {
                event_data["frame_height"] = ctx->frame->frame().height();
            }

            event("invoke", MA_OK, event_data);

            if (config_.websocket && ws_) {
                try {
                    // Build preview-compatible format (matches main branch):
                    // {"data": {"boxes":[[x,y,w,h,score,class_id],...], "labels":[...], "resolution":[w,h], "image":"..."}}
                    nlohmann::json preview_data = nlohmann::json::object();

                    // Transform boxes: object array → array-of-arrays (frontend compatibility)
                    if (event_data.contains("boxes") && event_data["boxes"].is_array()) {
                        nlohmann::json boxes_arr = nlohmann::json::array();
                        nlohmann::json labels_arr = nlohmann::json::array();
                        for (const auto& box : event_data["boxes"]) {
                            if (box.is_object()) {
                                double x = box.value("x", 0.0);
                                double y = box.value("y", 0.0);
                                double w = box.value("w", 0.0);
                                double h = box.value("h", 0.0);
                                double score = box.value("score", 0.0);
                                int cls = box.value("class_id", 0);
                                boxes_arr.push_back({x, y, w, h, score, cls});
                                labels_arr.push_back(box.value("label", ""));
                            }
                        }
                        preview_data["boxes"] = std::move(boxes_arr);
                        preview_data["labels"] = std::move(labels_arr);
                    }

                    // Resolution field - use preview_resolution config if set
                    int fw = config_.preview_width > 0 ? config_.preview_width : event_data.value("frame_width", 0);
                    int fh = config_.preview_height > 0 ? config_.preview_height : event_data.value("frame_height", 0);
                    preview_data["resolution"] = {fw, fh};

                    // Base64 JPEG image with configurable frame rate
                    try {
                        // Calculate frame interval from preview_fps
                        preview_interval_ms_ = 1000 / config_.preview_fps;

                        auto now_tp = std::chrono::steady_clock::now();
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now_tp - last_preview_time_).count();

                        if (elapsed_ms >= preview_interval_ms_) {
                            last_preview_time_ = now_tp;
                            cv::Mat mat = ctx->frame->frame().to_mat_copy();

                            if (!mat.empty()) {
                                // Apply preview resolution scaling if configured
                                if (config_.preview_width > 0 && config_.preview_height > 0) {
                                    if (mat.cols != config_.preview_width || mat.rows != config_.preview_height) {
                                        cv::Mat resized;
                                        cv::resize(mat, resized, cv::Size(config_.preview_width, config_.preview_height),
                                                   0, 0, cv::INTER_LINEAR);
                                        mat = std::move(resized);
                                    }
                                }

                                // to_mat_copy() already handles RGB→BGR conversion (see frame.cpp:234)
                                // No need to convert again here

                                // JPEG encoding with configurable quality
                                std::vector<uchar> jpeg_buf;
                                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config_.jpeg_quality};
                                if (cv::imencode(".jpg", mat, jpeg_buf, params)) {
                                    preview_data["image"] = base64_encode(jpeg_buf.data(), jpeg_buf.size());
                                } else {
                                    preview_data["image"] = "";
                                }
                            } else {
                                preview_data["image"] = "";
                            }
                        } else {
                            preview_data["image"] = "";  // Skip JPEG this frame (rate limit)
                        }
                    } catch (...) {
                        preview_data["image"] = "";
                    }

                    // Send simplified format (matches main branch frontend expectation)
                    nlohmann::json ws_msg = {{"data", std::move(preview_data)}};
                    std::string payload = ws_msg.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
                    if (ws_->broadcast_binary(reinterpret_cast<const uint8_t*>(payload.c_str()), payload.size())) {
                        ws_event_count_.fetch_add(1, std::memory_order_relaxed);
                    }
                } catch (const std::exception& e) {
                    if (config_.debug) {
                        std::cerr << "[ModelNode] websocket serialize/send failed: " << e.what() << "\n";
                    }
                }
            }

            // Forward to downstream
            forwardToDownstream(ctx, event_data);

        } catch (const std::exception& e) {
            error_count_.fetch_add(1, std::memory_order_relaxed);
            std::string error_msg = "Inference failed: " + std::string(e.what());
            event("error", MA_EIO, error_msg);
        }

        // Clean up: delete ctx (destructor calls frame->release())
        delete ctx;
    }
}

nlohmann::json ModelNode::runInference(const lua_cv::Frame& frame,
                                        const nlohmann::json& upstream) {
    if (config_.input_mode == FULL_FRAME) {
        return runFullFrameInference(frame, upstream);
    } else {
        return runCroppedRoiInference(frame, upstream);
    }
}

nlohmann::json ModelNode::runFullFrameInference(const lua_cv::Frame& frame,
                                                 const nlohmann::json& upstream) {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> output_shapes;
    PreprocessMeta preprocess_meta;
    auto t_total_start = std::chrono::steady_clock::now();
    auto t_pre_start = t_total_start;
    double preprocess_ms = 0.0;
    double vpss_ms = 0.0;
    double cpu_pre_ms = 0.0;
    double build_input_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double tpu_input_ms = 0.0;
    double tpu_forward_ms = 0.0;
    double tpu_output_ms = 0.0;
    bool vpss_attempted = false;
    std::string preprocess_path = "cpu";

#ifdef USE_CVI_TPU
    if (!session_) {
        throw std::runtime_error("CviSession not initialized");
    }

    int target_w = preprocess_config_.input_width > 0 ? preprocess_config_.input_width : frame.width();
    int target_h = preprocess_config_.input_height > 0 ? preprocess_config_.input_height : frame.height();

    preprocess_meta.ori_w = frame.width();
    preprocess_meta.ori_h = frame.height();
    preprocess_meta.input_w = target_w;
    preprocess_meta.input_h = target_h;

    bool use_vb = false;
    lua_cv::Frame preprocessed;
    std::string preprocess_type = to_lower(preprocess_config_.type);

#ifdef USE_CVI_MPI
    if (session_->supports_vb_input() &&
        frame.storage_type() == lua_cv::Frame::StorageType::CVI) {
        auto spec = session_->get_vb_input_spec();
        lua_cv::PixelFormat out_pf = lua_cv::from_cvi_pixel_format(spec.pixel_format);
        target_w = static_cast<int>(spec.width);
        target_h = static_cast<int>(spec.height);
        preprocess_meta.input_w = target_w;
        preprocess_meta.input_h = target_h;

        try {
            vpss_attempted = true;
            auto t_vpss_start = std::chrono::steady_clock::now();
            lua_cv::Frame work;
            if (frame.video_frame()) {
                work = lua_cv::Frame(*frame.video_frame(), false);
            } else {
                work = frame.clone();
            }

            lua_cv::CviVpssProcessor vpss;
            if (preprocess_type == "letterbox") {
                preprocess_meta = compute_letterbox_meta(frame.width(), frame.height(),
                                                         target_w, target_h,
                                                         preprocess_config_.center);
                vpss.letterbox(work, target_w, target_h,
                               static_cast<uint8_t>(preprocess_config_.fill_value),
                               nullptr, out_pf);
            } else if (preprocess_type == "resize" || preprocess_type == "none") {
                if (frame.width() != target_w || frame.height() != target_h ||
                    preprocess_type == "resize") {
                    vpss.resize(work, target_w, target_h);
                }
                preprocess_meta.scale = static_cast<float>(target_w) /
                                        static_cast<float>(std::max(1, frame.width()));
                preprocess_meta.pad_x = 0;
                preprocess_meta.pad_y = 0;
                preprocess_meta.ori_w = frame.width();
                preprocess_meta.ori_h = frame.height();
                preprocess_meta.input_w = target_w;
                preprocess_meta.input_h = target_h;
                if (work.pixel_format() != out_pf) {
                    vpss.convert_format(work, out_pf);
                }
            } else {
                throw std::runtime_error("Unsupported preprocess type: " + preprocess_config_.type);
            }

            preprocessed = std::move(work);
            std::string reason;
            if (lua_cv::cv_helpers::can_zero_copy(
                    preprocessed,
                    spec.pixel_format,
                    spec.width,
                    spec.height,
                    &reason)) {
                use_vb = true;
            }
            preprocess_path = use_vb ? "vpss_vb" : "vpss_copy";
            vpss_ms = elapsed_ms(t_vpss_start, std::chrono::steady_clock::now());
        } catch (const std::exception& e) {
            event("warning", 0, {{"message", "VPSS preprocess failed"}, {"detail", e.what()}});
            preprocessed = lua_cv::Frame();
        }
    }
#endif

    if (use_vb) {
#ifdef USE_CVI_MPI
        auto vb_mem = preprocessed.as_vb_memory();
        if (!vb_mem) {
            throw std::runtime_error("Failed to get VB memory from frame");
        }
        auto t_pre_end = std::chrono::steady_clock::now();
        preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
        session_->run_vb(vb_mem, &outputs, &output_shapes);
        auto t_infer_end = std::chrono::steady_clock::now();
        infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        const auto& stats = session_->last_run_stats();
        tpu_input_ms = stats.input_ms;
        tpu_forward_ms = stats.forward_ms;
        tpu_output_ms = stats.output_ms;
#endif
    } else {
        cv::Mat mat;
        bool skip_preprocess = false;
        if (!preprocessed.empty()) {
            mat = preprocessed.to_mat_copy();
            skip_preprocess = true;
        } else {
            mat = frame.to_mat_copy();
        }
        if (mat.empty()) {
            throw std::runtime_error("Frame is empty");
        }

        auto t_cpu_start = std::chrono::steady_clock::now();
        if (!skip_preprocess && preprocess_type == "letterbox") {
            if (target_w <= 0 || target_h <= 0) {
                target_w = mat.cols;
                target_h = mat.rows;
            }
            preprocess_meta = compute_letterbox_meta(mat.cols, mat.rows, target_w, target_h,
                                                     preprocess_config_.center);

            int new_w = static_cast<int>(std::floor(mat.cols * preprocess_meta.scale));
            int new_h = static_cast<int>(std::floor(mat.rows * preprocess_meta.scale));
            cv::Mat resized;
            if (new_w > 0 && new_h > 0 &&
                (new_w != mat.cols || new_h != mat.rows)) {
                cv::resize(mat, resized, cv::Size(new_w, new_h));
            } else {
                resized = mat;
            }

            int pad_w = target_w - new_w;
            int pad_h = target_h - new_h;
            int left = preprocess_config_.center ? pad_w / 2 : 0;
            int top = preprocess_config_.center ? pad_h / 2 : 0;
            int right = std::max(0, pad_w - left);
            int bottom = std::max(0, pad_h - top);

            cv::copyMakeBorder(resized, mat, top, bottom, left, right,
                               cv::BORDER_CONSTANT,
                               cv::Scalar(preprocess_config_.fill_value,
                                          preprocess_config_.fill_value,
                                          preprocess_config_.fill_value));
        } else if (!skip_preprocess && (preprocess_type == "resize" || preprocess_type == "none")) {
            if (target_w > 0 && target_h > 0 &&
                (mat.cols != target_w || mat.rows != target_h || preprocess_type == "resize")) {
                cv::resize(mat, mat, cv::Size(target_w, target_h));
            }
            preprocess_meta.scale = static_cast<float>(target_w) /
                                    static_cast<float>(std::max(1, frame.width()));
            preprocess_meta.pad_x = 0;
            preprocess_meta.pad_y = 0;
            preprocess_meta.ori_w = frame.width();
            preprocess_meta.ori_h = frame.height();
            preprocess_meta.input_w = target_w;
            preprocess_meta.input_h = target_h;
        } else if (!skip_preprocess) {
            throw std::runtime_error("Unsupported preprocess type: " + preprocess_config_.type);
        }
        cpu_pre_ms = elapsed_ms(t_cpu_start, std::chrono::steady_clock::now());

        std::vector<float> input_data;
        std::vector<int64_t> input_shape;
        auto t_build_start = std::chrono::steady_clock::now();
        build_float_input(mat, preprocess_config_, &input_data, &input_shape);
        build_input_ms = elapsed_ms(t_build_start, std::chrono::steady_clock::now());

        auto t_pre_end = std::chrono::steady_clock::now();
        preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
        session_->run_all(
            input_data.data(),
            input_shape,
            &outputs,
            &output_shapes);
        auto t_infer_end = std::chrono::steady_clock::now();
        infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        const auto& stats = session_->last_run_stats();
        tpu_input_ms = stats.input_ms;
        tpu_forward_ms = stats.forward_ms;
        tpu_output_ms = stats.output_ms;
    }
#else
    // CPU fallback - no TPU support
    (void)frame;
#endif

    // Build metadata for postprocess
    // Use Camera config resolution if available (for frontend coordinate mapping)
    // This ensures detection boxes are scaled to match the frontend video stream
    int meta_frame_w = frame.width();
    int meta_frame_h = frame.height();
    if (upstream_camera_) {
        meta_frame_w = upstream_camera_->config_width();
        meta_frame_h = upstream_camera_->config_height();
    }

    nlohmann::json meta = {
        {"upstream", upstream},
        {"threshold", config_.conf_threshold},
        {"frame_width", meta_frame_w},
        {"frame_height", meta_frame_h},
        {"output_count", session_ ? session_->output_count() : 1}
    };
    fill_meta_json(preprocess_meta, &meta);

    auto t_post_start = std::chrono::steady_clock::now();
    nlohmann::json result = callPostprocess(std::move(outputs), std::move(output_shapes), meta);
    auto t_post_end = std::chrono::steady_clock::now();
    postprocess_ms = elapsed_ms(t_post_start, t_post_end);

    if (config_.profile) {
        nlohmann::json profile = {
            {"mode", "full_frame"},
            {"preprocess_ms", preprocess_ms},
            {"preprocess_vpss_ms", vpss_ms},
            {"preprocess_cpu_ms", cpu_pre_ms},
            {"build_input_ms", build_input_ms},
            {"infer_ms", infer_ms},
            {"postprocess_ms", postprocess_ms},
            {"tpu_input_ms", tpu_input_ms},
            {"tpu_forward_ms", tpu_forward_ms},
            {"tpu_output_ms", tpu_output_ms},
            {"preprocess_path", preprocess_path},
            {"vpss_attempted", vpss_attempted},
            {"use_vb", use_vb},
            {"zero_copy", use_vb},
            {"input_w", preprocess_meta.input_w},
            {"input_h", preprocess_meta.input_h}
        };
        profile["total_ms"] = elapsed_ms(t_total_start, t_post_end);
        emitProfile(profile);
    }

    return result;
}

nlohmann::json ModelNode::runCroppedRoiInference(const lua_cv::Frame& frame,
                                                  const nlohmann::json& upstream) {
    if (!select_rois_.isFunction()) {
        return {{"items", nlohmann::json::array()}};
    }

    LuaIntf::LuaRef upstream_ref = json_to_luaref(L_, upstream);
    LuaIntf::LuaRef rois_ref = select_rois_.call<LuaIntf::LuaRef>(upstream_ref);
    nlohmann::json rois_json = luaref_to_json(rois_ref);

    if (!rois_json.is_array() || rois_json.empty()) {
        return {{"items", nlohmann::json::array()}};
    }

    nlohmann::json items = nlohmann::json::array();
    auto t_total_start = std::chrono::steady_clock::now();
    double preprocess_total_ms = 0.0;
    double vpss_total_ms = 0.0;
    double cpu_pre_total_ms = 0.0;
    double build_input_total_ms = 0.0;
    double infer_total_ms = 0.0;
    double postprocess_total_ms = 0.0;
    double tpu_input_total_ms = 0.0;
    double tpu_forward_total_ms = 0.0;
    double tpu_output_total_ms = 0.0;
    int roi_count = 0;
    int use_vb_count = 0;
    int vpss_attempted_count = 0;

#ifdef USE_CVI_MPI
    std::unique_ptr<lua_cv::CviVpssProcessor> vpss;
    if (session_ && session_->supports_vb_input() &&
        frame.storage_type() == lua_cv::Frame::StorageType::CVI) {
        vpss = std::make_unique<lua_cv::CviVpssProcessor>();
    }
#endif

    for (const auto& roi_item : rois_json) {
        Roi roi;
        if (!parse_roi(roi_item, &roi)) {
            continue;
        }

        int x = roi.x;
        int y = roi.y;
        int w = roi.w;
        int h = roi.h;

        // Clamp ROI to frame bounds
        x = std::max(0, std::min(x, frame.width() - 1));
        y = std::max(0, std::min(y, frame.height() - 1));
        w = std::min(w, frame.width() - x);
        h = std::min(h, frame.height() - y);

        if (w <= 0 || h <= 0) {
            continue;
        }

        auto t_pre_start = std::chrono::steady_clock::now();
        double vpss_ms_local = 0.0;
        double cpu_pre_ms_local = 0.0;
        double build_input_ms_local = 0.0;
        double infer_ms_local = 0.0;
        double postprocess_ms_local = 0.0;
        bool vpss_attempted_local = false;

        int target_w = config_.crop_size_explicit ? config_.crop_width : preprocess_config_.input_width;
        int target_h = config_.crop_size_explicit ? config_.crop_height : preprocess_config_.input_height;
        if (target_w <= 0 || target_h <= 0) {
            target_w = w;
            target_h = h;
        }

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int64_t>> output_shapes;
        PreprocessMeta preprocess_meta;
        preprocess_meta.ori_w = w;
        preprocess_meta.ori_h = h;
        preprocess_meta.input_w = target_w;
        preprocess_meta.input_h = target_h;

#ifdef USE_CVI_TPU
        if (!session_) {
            throw std::runtime_error("CviSession not initialized");
        }

        bool use_vb = false;
        lua_cv::Frame preprocessed;
        std::string preprocess_type = to_lower(preprocess_config_.type);

#ifdef USE_CVI_MPI
        if (vpss) {
            auto spec = session_->get_vb_input_spec();
            lua_cv::PixelFormat out_pf = lua_cv::from_cvi_pixel_format(spec.pixel_format);
            target_w = static_cast<int>(spec.width);
            target_h = static_cast<int>(spec.height);
            preprocess_meta.input_w = target_w;
            preprocess_meta.input_h = target_h;

            try {
                vpss_attempted_local = true;
                vpss_attempted_count++;
                auto t_vpss_start = std::chrono::steady_clock::now();
                lua_cv::Frame work;
                if (frame.video_frame()) {
                    work = lua_cv::Frame(*frame.video_frame(), false);
                } else {
                    work = frame.clone();
                }

                if (preprocess_type == "letterbox") {
                    vpss->crop(work, x, y, w, h);
                    preprocess_meta = compute_letterbox_meta(w, h, target_w, target_h,
                                                             preprocess_config_.center);
                    vpss->letterbox(work, target_w, target_h,
                                    static_cast<uint8_t>(preprocess_config_.fill_value),
                                    nullptr, out_pf);
                } else if (preprocess_type == "resize" || preprocess_type == "none") {
                    if (w != target_w || h != target_h || preprocess_type == "resize") {
                        vpss->crop_resize(work, x, y, w, h,
                                          target_w, target_h,
                                          out_pf);
                    } else {
                        vpss->crop(work, x, y, w, h);
                        if (work.pixel_format() != out_pf) {
                            vpss->convert_format(work, out_pf);
                        }
                    }
                    preprocess_meta.scale = static_cast<float>(target_w) /
                                            static_cast<float>(std::max(1, w));
                    preprocess_meta.pad_x = 0;
                    preprocess_meta.pad_y = 0;
                    preprocess_meta.ori_w = w;
                    preprocess_meta.ori_h = h;
                    preprocess_meta.input_w = target_w;
                    preprocess_meta.input_h = target_h;
                } else {
                    throw std::runtime_error("Unsupported preprocess type: " + preprocess_config_.type);
                }

                preprocessed = std::move(work);
                std::string reason;
                if (lua_cv::cv_helpers::can_zero_copy(
                        preprocessed,
                        spec.pixel_format,
                        spec.width,
                        spec.height,
                        &reason)) {
                    use_vb = true;
                }
                if (use_vb) {
                    use_vb_count++;
                }
                vpss_ms_local = elapsed_ms(t_vpss_start, std::chrono::steady_clock::now());
            } catch (const std::exception& e) {
                event("warning", 0, {{"message", "VPSS ROI preprocess failed"}, {"detail", e.what()}});
                preprocessed = lua_cv::Frame();
            }
        }
#endif

        if (use_vb) {
#ifdef USE_CVI_MPI
            auto vb_mem = preprocessed.as_vb_memory();
            if (!vb_mem) {
                throw std::runtime_error("Failed to get VB memory from ROI frame");
            }
            auto t_pre_end = std::chrono::steady_clock::now();
            double preprocess_ms_local = elapsed_ms(t_pre_start, t_pre_end);
            auto t_infer_start = std::chrono::steady_clock::now();
            session_->run_vb(vb_mem, &outputs, &output_shapes);
            auto t_infer_end = std::chrono::steady_clock::now();
            infer_ms_local = elapsed_ms(t_infer_start, t_infer_end);
            const auto& stats = session_->last_run_stats();
            tpu_input_total_ms += stats.input_ms;
            tpu_forward_total_ms += stats.forward_ms;
            tpu_output_total_ms += stats.output_ms;
            preprocess_total_ms += preprocess_ms_local;
#endif
        } else {
            cv::Mat mat;
            bool skip_preprocess = false;
            if (!preprocessed.empty()) {
                mat = preprocessed.to_mat_copy();
                skip_preprocess = true;
            } else {
                cv::Mat src = frame.to_mat_copy();
                if (src.empty()) {
                    continue;
                }

                cv::Rect roi_rect(x, y, w, h);
                mat = src(roi_rect).clone();
            }

            auto t_cpu_start = std::chrono::steady_clock::now();
            if (!skip_preprocess && preprocess_type == "letterbox") {
                preprocess_meta = compute_letterbox_meta(w, h, target_w, target_h,
                                                         preprocess_config_.center);
                int new_w = static_cast<int>(std::floor(w * preprocess_meta.scale));
                int new_h = static_cast<int>(std::floor(h * preprocess_meta.scale));
                cv::Mat resized;
                if (new_w > 0 && new_h > 0 &&
                    (new_w != w || new_h != h)) {
                    cv::resize(mat, resized, cv::Size(new_w, new_h));
                } else {
                    resized = mat;
                }

                int pad_w = target_w - new_w;
                int pad_h = target_h - new_h;
                int left = preprocess_config_.center ? pad_w / 2 : 0;
                int top = preprocess_config_.center ? pad_h / 2 : 0;
                int right = std::max(0, pad_w - left);
                int bottom = std::max(0, pad_h - top);

                cv::copyMakeBorder(resized, mat, top, bottom, left, right,
                                   cv::BORDER_CONSTANT,
                                   cv::Scalar(preprocess_config_.fill_value,
                                              preprocess_config_.fill_value,
                                              preprocess_config_.fill_value));
            } else if (!skip_preprocess &&
                       (preprocess_type == "resize" || preprocess_type == "none")) {
                if (mat.cols != target_w || mat.rows != target_h || preprocess_type == "resize") {
                    cv::resize(mat, mat, cv::Size(target_w, target_h));
                }
                preprocess_meta.scale = static_cast<float>(target_w) /
                                        static_cast<float>(std::max(1, w));
                preprocess_meta.pad_x = 0;
                preprocess_meta.pad_y = 0;
                preprocess_meta.ori_w = w;
                preprocess_meta.ori_h = h;
                preprocess_meta.input_w = target_w;
                preprocess_meta.input_h = target_h;
            } else if (!skip_preprocess) {
                throw std::runtime_error("Unsupported preprocess type: " + preprocess_config_.type);
            }
            cpu_pre_ms_local = elapsed_ms(t_cpu_start, std::chrono::steady_clock::now());

            std::vector<float> input_data;
            std::vector<int64_t> input_shape;
            auto t_build_start = std::chrono::steady_clock::now();
            build_float_input(mat, preprocess_config_, &input_data, &input_shape);
            build_input_ms_local = elapsed_ms(t_build_start, std::chrono::steady_clock::now());

            auto t_pre_end = std::chrono::steady_clock::now();
            double preprocess_ms_local = elapsed_ms(t_pre_start, t_pre_end);
            auto t_infer_start = std::chrono::steady_clock::now();
            session_->run_all(
                input_data.data(),
                input_shape,
                &outputs,
                &output_shapes);
            auto t_infer_end = std::chrono::steady_clock::now();
            infer_ms_local = elapsed_ms(t_infer_start, t_infer_end);
            const auto& stats = session_->last_run_stats();
            tpu_input_total_ms += stats.input_ms;
            tpu_forward_total_ms += stats.forward_ms;
            tpu_output_total_ms += stats.output_ms;
            preprocess_total_ms += preprocess_ms_local;
        }
#else
        (void)frame;
#endif

        nlohmann::json meta = {
            {"roi", {{"x", x}, {"y", y}, {"w", w}, {"h", h}}},
            {"upstream", upstream},
            {"threshold", config_.conf_threshold}
        };
        fill_meta_json(preprocess_meta, &meta);

        auto t_post_start = std::chrono::steady_clock::now();
        auto item = callPostprocess(std::move(outputs), std::move(output_shapes), meta);
        auto t_post_end = std::chrono::steady_clock::now();
        postprocess_ms_local = elapsed_ms(t_post_start, t_post_end);
        postprocess_total_ms += postprocess_ms_local;
        vpss_total_ms += vpss_ms_local;
        cpu_pre_total_ms += cpu_pre_ms_local;
        build_input_total_ms += build_input_ms_local;
        infer_total_ms += infer_ms_local;
        roi_count++;
        items.push_back(item);
    }

    if (config_.profile && roi_count > 0) {
        nlohmann::json profile = {
            {"mode", "cropped_roi"},
            {"roi_count", roi_count},
            {"preprocess_ms", preprocess_total_ms},
            {"preprocess_vpss_ms", vpss_total_ms},
            {"preprocess_cpu_ms", cpu_pre_total_ms},
            {"build_input_ms", build_input_total_ms},
            {"infer_ms", infer_total_ms},
            {"postprocess_ms", postprocess_total_ms},
            {"tpu_input_ms", tpu_input_total_ms},
            {"tpu_forward_ms", tpu_forward_total_ms},
            {"tpu_output_ms", tpu_output_total_ms},
            {"use_vb_count", use_vb_count},
            {"vpss_attempted_count", vpss_attempted_count}
        };
        profile["total_ms"] = elapsed_ms(t_total_start, std::chrono::steady_clock::now());
        profile["avg_preprocess_ms"] = preprocess_total_ms / roi_count;
        profile["avg_infer_ms"] = infer_total_ms / roi_count;
        profile["avg_postprocess_ms"] = postprocess_total_ms / roi_count;
        emitProfile(profile);
    }

    return {{"items", items}};
}

nlohmann::json ModelNode::callPostprocess(
    std::vector<std::vector<float>> outputs,
    std::vector<std::vector<int64_t>> output_shapes,
    const nlohmann::json& meta) {
    try {
        if (outputs.size() != output_shapes.size()) {
            event("warning", 0, {
                {"message", "Output count mismatch"},
                {"outputs", outputs.size()},
                {"shapes", output_shapes.size()}
            });
        }

        LuaIntf::LuaRef outputs_ref = LuaIntf::LuaRef::createTable(L_);
        const std::vector<std::string>* output_names = nullptr;
#ifdef USE_CVI_TPU
        if (session_) {
            output_names = &session_->get_output_names();
        }
#endif

        size_t count = std::min(outputs.size(), output_shapes.size());
        for (size_t i = 0; i < count; ++i) {
            tensor::Tensor tensor(std::move(outputs[i]), output_shapes[i]);
            std::string default_name = "output" + std::to_string(i);
            outputs_ref[default_name] = tensor;
            if (output_names && i < output_names->size()) {
                const std::string& name = (*output_names)[i];
                if (!name.empty() && name != default_name) {
                    outputs_ref[name] = tensor;
                }
            }
        }

        LuaIntf::LuaRef meta_ref = json_to_luaref(L_, meta);

        // Call postprocess(outputs, meta)
        LuaIntf::LuaRef result = postprocess_.call<LuaIntf::LuaRef>(outputs_ref, meta_ref);
        nlohmann::json json_result = luaref_to_json(result);

        // Clear LuaRef to prevent memory leak
        result = LuaIntf::LuaRef();
        outputs_ref = LuaIntf::LuaRef();
        meta_ref = LuaIntf::LuaRef();
        // Keep GC moving; large tensor outputs can otherwise accumulate.
        if (L_) {
            lua_gc(L_, LUA_GCSTEP, 200);
        }

        return json_result;

    } catch (const LuaIntf::LuaException& e) {
        std::string error_msg = "Postprocess Lua error: " + std::string(e.what());
        event("error", MA_EINVAL, error_msg);
        return nlohmann::json::object();

    } catch (const std::exception& e) {
        std::string error_msg = "Postprocess error: " + std::string(e.what());
        event("error", MA_EINVAL, error_msg);
        return nlohmann::json::object();
    }
}

void ModelNode::forwardToDownstream(PipelineContext* ctx, const nlohmann::json& result) {
    if (downstream_.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(subscribers_mutex_);
    for (auto* mbox : downstream_) {
        // Increase reference count before creating new PipelineContext
        ctx->frame->ref();

        auto* next_ctx = new PipelineContext{
            ctx->frame,
            result,
            ctx->frame_id
        };

        if (!mbox->post(next_ctx, 0)) {
            // Queue full, cleanup
            delete next_ctx;
        }
    }
}

void ModelNode::updateEwma(double elapsed_ms) {
    infer_ema_ms_ = kEmaAlpha * elapsed_ms + (1.0 - kEmaAlpha) * infer_ema_ms_;
}

void ModelNode::emitProfile(const nlohmann::json& profile) {
    if (!config_.profile) {
        return;
    }

    nlohmann::json payload = profile;
    payload["node_id"] = id_;
    payload["node_type"] = type_;

    if (server_) {
        event("profile", 0, payload);
    }

    std::cerr << "[PROFILE] " << payload.dump() << "\n";
}

void ModelNode::cleanupLuaRef() {
    // Must clear LuaRef before closing State
    postprocess_ = LuaIntf::LuaRef();
    select_rois_ = LuaIntf::LuaRef();
    preprocess_config_ref_ = LuaIntf::LuaRef();

    if (L_) {
        lua_close(L_);
        L_ = nullptr;
    }
}

} // namespace node
