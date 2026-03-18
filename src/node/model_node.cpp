#include "model_node.h"
#include "node_factory.h"
#include "node_server.h"
#include "camera_node.h"
#include "lua_model_postprocess.h"
#include "lua_roi_selector.h"
#include "luaref_json_bridge.h"
#include "model_event_payload.h"
#include "model_inference_meta.h"
#include "model_node_utils.h"
#include "model_profile_payload.h"
#include "resource_estimator.h"
#include "stream/websocket_transport.h"
#include "stream/model_preview_formatter.h"
#include "modules/cv/cv_helpers.h"
#include "modules/cv/cv_types.h"
#include "tensor/tensor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

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
}  // namespace

// Register ModelNode type
REGISTER_NODE("model", ModelNode);

ModelNode::ModelNode(const std::string& id, const std::string& type)
    : DataNode(id, type, 1) {}

ModelNode::~ModelNode() {
    onDestroy();
}

int ModelNode::parseConfig(const nlohmann::json& config) {
    if (config.contains("model")) {
        config_.model_path = config.at("model");
    } else if (config.contains("uri")) {
        config_.model_path = config.at("uri");
    } else {
        config_.model_path = "/usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel";
    }

#ifdef USE_CVI_TPU
    if (!file_exists(config_.model_path)) {
        last_error_ = "Model file not found: " + config_.model_path;
        return MA_ENOENT;
    }
#endif

    if (config.contains("script")) {
        config_.script_path = config.at("script");
    } else {
        config_.script_path = "/userdata/scripts/yolo11_tensor_detector.lua";
    }

    if (!file_exists(config_.script_path)) {
        last_error_ = "Script file not found: " + config_.script_path;
        return MA_ENOENT;
    }

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

    parsePreviewConfig(config);
    return MA_OK;
}

void ModelNode::parsePreviewConfig(const nlohmann::json& config) {
    if (config.contains("preview_resolution") && config["preview_resolution"].is_string()) {
        config_.preview_resolution = config["preview_resolution"].get<std::string>();
        if (parse_preview_resolution(config_.preview_resolution, &config_.preview_width, &config_.preview_height)) {
            std::cout << "[ModelNode] Preview resolution: " << config_.preview_width
                      << "x" << config_.preview_height << std::endl;
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
}

void ModelNode::configureLuaPathFromScript() const {
    const char* existing_lua_path = std::getenv("LUA_PATH");
    if (existing_lua_path) {
        return;
    }

    size_t last_slash = config_.script_path.find_last_of("/\\");
    if (last_slash == std::string::npos) {
        return;
    }

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

int ModelNode::initializeLuaRuntime() {
    L_ = luaL_newstate();
    if (!L_) {
        last_error_ = "Failed to create Lua state";
        return MA_ENOMEM;
    }
    luaL_openlibs(L_);

    lua_cv::register_module(L_);
    lua_nn::register_module(L_);
    lua_utils::register_module(L_);
    return MA_OK;
}

int ModelNode::loadLuaModelBindings() {
    if (luaL_dofile(L_, config_.script_path.c_str()) != LUA_OK) {
        std::string err = lua_tostring(L_, -1);
        last_error_ = "Lua script error: " + err;
        cleanupLuaRef();
        return MA_EINVAL;
    }

    LuaIntf::LuaRef model = LuaIntf::LuaRef::popFromStack(L_);
    if (!model.isTable()) {
        last_error_ = "Script must return a table";
        model = LuaIntf::LuaRef();
        cleanupLuaRef();
        return MA_EINVAL;
    }

    postprocess_ = model["postprocess"];
    select_rois_ = model["select_rois"];
    preprocess_config_ref_ = model["preprocess_config"];

    if (!postprocess_.isFunction()) {
        last_error_ = "Missing postprocess function in script";
        model = LuaIntf::LuaRef();
        cleanupLuaRef();
        return MA_EINVAL;
    }

    if (preprocess_config_ref_.isTable()) {
        preprocess_config_ = PreprocessConfig::fromLuaRef(preprocess_config_ref_);
        preprocess_config_explicit_ = true;
    }

    model = LuaIntf::LuaRef();
    return MA_OK;
}

int ModelNode::initializeSession() {
#ifdef USE_CVI_TPU
    try {
        session_ = std::make_unique<inference::CviSession>(config_.model_path);

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
        cleanupLuaRef();
        return MA_EIO;
    }
#else
    FILE* f = fopen(config_.model_path.c_str(), "r");
    if (!f) {
        last_error_ = "Model file not found: " + config_.model_path;
        cleanupLuaRef();
        return MA_ENOENT;
    }
    fclose(f);
#endif
    return MA_OK;
}

int ModelNode::validateInputModeDependencies() {
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
            cleanupLuaRef();
            return MA_EINVAL;
        }
        if (!select_rois_.isFunction()) {
            last_error_ = "CROPPED_ROI mode requires select_rois(upstream) in script";
            cleanupLuaRef();
            return MA_EINVAL;
        }
    }
    return MA_OK;
}

void ModelNode::bindUpstreamCamera() {
    upstream_camera_ = nullptr;
    for (const auto& [dep_id, dep] : dependencies_) {
        if (dep->type() == "camera") {
            upstream_camera_ = static_cast<CameraNode*>(dep);
            break;
        }
    }
}

int ModelNode::onCreate(const nlohmann::json& config) {
    int ret = parseConfig(config);
    if (ret != MA_OK) {
        return ret;
    }

    configureLuaPathFromScript();

    ret = initializeLuaRuntime();
    if (ret != MA_OK) {
        return ret;
    }

    ret = loadLuaModelBindings();
    if (ret != MA_OK) {
        return ret;
    }

    ret = initializeSession();
    if (ret != MA_OK) {
        return ret;
    }

    ret = validateInputModeDependencies();
    if (ret != MA_OK) {
        return ret;
    }

    bindUpstreamCamera();
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

void ModelNode::maybeBroadcastPreview(const PipelineContext& ctx, const nlohmann::json& event_data) {
    if (!config_.websocket || !ws_) {
        return;
    }

    try {
        ModelPreviewFormatConfig preview_config;
        preview_config.preview_width = config_.preview_width;
        preview_config.preview_height = config_.preview_height;
        preview_config.preview_fps = config_.preview_fps;
        preview_config.jpeg_quality = config_.jpeg_quality;

        nlohmann::json ws_msg = build_model_preview_message(
            event_data,
            ctx.frame->frame(),
            preview_config,
            &last_preview_time_,
            &preview_interval_ms_);
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

            nlohmann::json event_data = build_inference_event_data(
                std::move(result),
                ctx->frame_id,
                ctx->frame->frame().width(),
                ctx->frame->frame().height());
            event_data["latency_ms"] = elapsed_ms;

            event("invoke", MA_OK, event_data);
            maybeBroadcastPreview(*ctx, event_data);

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
    InferenceTimings timings;

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
            timings.vpss_attempted = true;
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
            timings.preprocess_path = use_vb ? "vpss_vb" : "vpss_copy";
            timings.vpss_ms = elapsed_ms(t_vpss_start, std::chrono::steady_clock::now());
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
        timings.preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
        session_->run_vb(vb_mem, &outputs, &output_shapes);
        auto t_infer_end = std::chrono::steady_clock::now();
        timings.infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        const auto& stats = session_->last_run_stats();
        timings.tpu_input_ms = stats.input_ms;
        timings.tpu_forward_ms = stats.forward_ms;
        timings.tpu_output_ms = stats.output_ms;
        timings.use_vb = true;
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
        timings.cpu_pre_ms = elapsed_ms(t_cpu_start, std::chrono::steady_clock::now());

        std::vector<float> input_data;
        std::vector<int64_t> input_shape;
        auto t_build_start = std::chrono::steady_clock::now();
        build_float_input(mat, preprocess_config_, &input_data, &input_shape);
        timings.build_input_ms = elapsed_ms(t_build_start, std::chrono::steady_clock::now());

        auto t_pre_end = std::chrono::steady_clock::now();
        timings.preprocess_ms = elapsed_ms(t_pre_start, t_pre_end);
        auto t_infer_start = std::chrono::steady_clock::now();
        session_->run_all(
            input_data.data(),
            input_shape,
            &outputs,
            &output_shapes);
        auto t_infer_end = std::chrono::steady_clock::now();
        timings.infer_ms = elapsed_ms(t_infer_start, t_infer_end);
        const auto& stats = session_->last_run_stats();
        timings.tpu_input_ms = stats.input_ms;
        timings.tpu_forward_ms = stats.forward_ms;
        timings.tpu_output_ms = stats.output_ms;
    }
#else
    // CPU fallback - no TPU support
    (void)frame;
#endif

    int meta_frame_w = frame.width();
    int meta_frame_h = frame.height();
    if (upstream_camera_) {
        meta_frame_w = upstream_camera_->config_width();
        meta_frame_h = upstream_camera_->config_height();
    }
    nlohmann::json meta = build_full_frame_meta(
        upstream,
        config_.conf_threshold,
        meta_frame_w,
        meta_frame_h,
        session_ ? session_->output_count() : 1,
        preprocess_meta);

    auto t_post_start = std::chrono::steady_clock::now();
    nlohmann::json result = callPostprocess(std::move(outputs), std::move(output_shapes), meta);
    auto t_post_end = std::chrono::steady_clock::now();
    timings.postprocess_ms = elapsed_ms(t_post_start, t_post_end);
    if (config_.profile) {
        emitProfile(build_full_frame_profile_payload(
            timings, preprocess_meta, elapsed_ms(t_total_start, t_post_end)));
    }

    return result;
}

nlohmann::json ModelNode::runCroppedRoiInference(const lua_cv::Frame& frame,
                                                  const nlohmann::json& upstream) {
    std::vector<Roi> rois = select_valid_rois(
        L_, select_rois_, frame.width(), frame.height(), upstream);
    if (rois.empty()) {
        return {{"items", nlohmann::json::array()}};
    }

    nlohmann::json items = nlohmann::json::array();
    auto t_total_start = std::chrono::steady_clock::now();
    RoiBatchMetrics metrics;
    for (const auto& roi : rois) {
        nlohmann::json item = runSingleRoiInference(frame, roi, upstream, &metrics);
        if (!item.is_null()) {
            items.push_back(std::move(item));
        }
    }

    if (config_.profile && metrics.roi_count > 0) {
        emitProfile(build_cropped_roi_profile_payload(
            metrics, elapsed_ms(t_total_start, std::chrono::steady_clock::now())));
    }
    return {{"items", items}};
}

nlohmann::json ModelNode::runSingleRoiInference(const lua_cv::Frame& frame,
                                                const Roi& roi,
                                                const nlohmann::json& upstream,
                                                RoiBatchMetrics* metrics) {
    auto t_pre_start = std::chrono::steady_clock::now();
    double vpss_ms_local = 0.0;
    double cpu_pre_ms_local = 0.0;
    double build_input_ms_local = 0.0;
    double infer_ms_local = 0.0;
    double postprocess_ms_local = 0.0;

    int target_w = config_.crop_size_explicit ? config_.crop_width : preprocess_config_.input_width;
    int target_h = config_.crop_size_explicit ? config_.crop_height : preprocess_config_.input_height;
    if (target_w <= 0 || target_h <= 0) {
        target_w = roi.w;
        target_h = roi.h;
    }

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> output_shapes;
    PreprocessMeta preprocess_meta;
    preprocess_meta.ori_w = roi.w;
    preprocess_meta.ori_h = roi.h;
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
    if (session_->supports_vb_input() &&
        frame.storage_type() == lua_cv::Frame::StorageType::CVI) {
        auto spec = session_->get_vb_input_spec();
        lua_cv::PixelFormat out_pf = lua_cv::from_cvi_pixel_format(spec.pixel_format);
        target_w = static_cast<int>(spec.width);
        target_h = static_cast<int>(spec.height);
        preprocess_meta.input_w = target_w;
        preprocess_meta.input_h = target_h;

        try {
            if (metrics) {
                metrics->vpss_attempted_count++;
            }
            auto t_vpss_start = std::chrono::steady_clock::now();
            lua_cv::Frame work;
            if (frame.video_frame()) {
                work = lua_cv::Frame(*frame.video_frame(), false);
            } else {
                work = frame.clone();
            }

            lua_cv::CviVpssProcessor vpss;
            if (preprocess_type == "letterbox") {
                vpss.crop(work, roi.x, roi.y, roi.w, roi.h);
                preprocess_meta = compute_letterbox_meta(roi.w, roi.h, target_w, target_h,
                                                         preprocess_config_.center);
                vpss.letterbox(work, target_w, target_h,
                               static_cast<uint8_t>(preprocess_config_.fill_value),
                               nullptr, out_pf);
            } else if (preprocess_type == "resize" || preprocess_type == "none") {
                if (roi.w != target_w || roi.h != target_h || preprocess_type == "resize") {
                    vpss.crop_resize(work, roi.x, roi.y, roi.w, roi.h,
                                     target_w, target_h,
                                     out_pf);
                } else {
                    vpss.crop(work, roi.x, roi.y, roi.w, roi.h);
                    if (work.pixel_format() != out_pf) {
                        vpss.convert_format(work, out_pf);
                    }
                }
                preprocess_meta.scale = static_cast<float>(target_w) /
                                        static_cast<float>(std::max(1, roi.w));
                preprocess_meta.pad_x = 0;
                preprocess_meta.pad_y = 0;
                preprocess_meta.ori_w = roi.w;
                preprocess_meta.ori_h = roi.h;
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
            if (use_vb && metrics) {
                metrics->use_vb_count++;
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
        if (metrics) {
            metrics->tpu_input_total_ms += stats.input_ms;
            metrics->tpu_forward_total_ms += stats.forward_ms;
            metrics->tpu_output_total_ms += stats.output_ms;
            metrics->preprocess_total_ms += preprocess_ms_local;
        }
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
                return nullptr;
            }

            cv::Rect roi_rect(roi.x, roi.y, roi.w, roi.h);
            mat = src(roi_rect).clone();
        }

        auto t_cpu_start = std::chrono::steady_clock::now();
        if (!skip_preprocess && preprocess_type == "letterbox") {
            preprocess_meta = compute_letterbox_meta(roi.w, roi.h, target_w, target_h,
                                                     preprocess_config_.center);
            int new_w = static_cast<int>(std::floor(roi.w * preprocess_meta.scale));
            int new_h = static_cast<int>(std::floor(roi.h * preprocess_meta.scale));
            cv::Mat resized;
            if (new_w > 0 && new_h > 0 &&
                (new_w != roi.w || new_h != roi.h)) {
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
                                    static_cast<float>(std::max(1, roi.w));
            preprocess_meta.pad_x = 0;
            preprocess_meta.pad_y = 0;
            preprocess_meta.ori_w = roi.w;
            preprocess_meta.ori_h = roi.h;
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
        if (metrics) {
            metrics->tpu_input_total_ms += stats.input_ms;
            metrics->tpu_forward_total_ms += stats.forward_ms;
            metrics->tpu_output_total_ms += stats.output_ms;
            metrics->preprocess_total_ms += preprocess_ms_local;
        }
    }
#else
    (void)frame;
#endif

    nlohmann::json meta = build_roi_meta(roi, upstream, config_.conf_threshold, preprocess_meta);
    auto t_post_start = std::chrono::steady_clock::now();
    auto item = callPostprocess(std::move(outputs), std::move(output_shapes), meta);
    auto t_post_end = std::chrono::steady_clock::now();
    postprocess_ms_local = elapsed_ms(t_post_start, t_post_end);

    if (metrics) {
        metrics->postprocess_total_ms += postprocess_ms_local;
        metrics->vpss_total_ms += vpss_ms_local;
        metrics->cpu_pre_total_ms += cpu_pre_ms_local;
        metrics->build_input_total_ms += build_input_ms_local;
        metrics->infer_total_ms += infer_ms_local;
        metrics->roi_count++;
    }
    return item;
}

nlohmann::json ModelNode::callPostprocess(
    std::vector<std::vector<float>> outputs,
    std::vector<std::vector<int64_t>> output_shapes,
    const nlohmann::json& meta) {
    const std::vector<std::string>* output_names = nullptr;
#ifdef USE_CVI_TPU
    if (session_) {
        output_names = &session_->get_output_names();
    }
#endif

    LuaModelPostprocessResult result = call_lua_model_postprocess(
        L_, postprocess_, output_names, std::move(outputs), std::move(output_shapes), meta);
    if (result.has_warning) {
        event("warning", 0, result.warning_payload);
    }
    if (result.has_error) {
        event("error", MA_EINVAL, result.error_message);
    }
    return std::move(result.value);
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
