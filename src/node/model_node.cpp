#include "model_node.h"
#include "node_factory.h"
#include "node_server.h"
#include "camera_node.h"
#include "lua_model_postprocess.h"
#include "lua_roi_selector.h"
#include "luaref_json_bridge.h"
#include "model_event_payload.h"
#include "model_inference_executor.h"
#include "model_inference_meta.h"
#include "model_lua_runtime.h"
#include "model_node_utils.h"
#include "model_profile_payload.h"
#include "resource_estimator.h"
#include "stream/websocket_transport.h"
#include "stream/model_preview_formatter.h"
#include "inference/layout.h"
#include "modules/cv/mmf_context.h"

#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include <lua.h>

#ifdef USE_CVI_TPU
#include "inference/cvi_session.h"
#endif

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
    std::string preview_resolution_key = "previewResolution";

    if (config.contains(preview_resolution_key) && config[preview_resolution_key].is_string()) {
        config_.preview_resolution = config[preview_resolution_key].get<std::string>();
        if (parse_preview_resolution(config_.preview_resolution, &config_.preview_width, &config_.preview_height)) {
            std::cout << "[ModelNode] Preview resolution: " << config_.preview_width
                      << "x" << config_.preview_height << std::endl;
        } else {
            std::cerr << "[ModelNode] Invalid preview_resolution format (expected WIDTHxHEIGHT), using original size" << std::endl;
            config_.preview_width = 0;
            config_.preview_height = 0;
        }
    }

    if (config.contains("previewFps") && config["previewFps"].is_number()) {
        config_.preview_fps = config["previewFps"].get<int>();
        if (config_.preview_fps <= 0 || config_.preview_fps > 30) {
            std::cerr << "[ModelNode] Invalid preview_fps, using default 15" << std::endl;
            config_.preview_fps = 15;
        }
    }
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

    configure_lua_path_from_script(config_.script_path);

    LuaRuntimeInitResult runtime = create_model_lua_runtime();
    if (runtime.code != MA_OK) {
        last_error_ = runtime.error_message;
        return runtime.code;
    }
    L_ = runtime.state;

    ModelScriptLoadResult bindings = load_model_script_bindings(L_, config_.script_path);
    if (bindings.code != MA_OK) {
        last_error_ = bindings.error_message;
        cleanupLuaRef();
        return bindings.code;
    }
    postprocess_ = bindings.bindings.postprocess;
    select_rois_ = bindings.bindings.select_rois;
    preprocess_config_ref_ = bindings.bindings.preprocess_config_ref;
    preprocess_config_ = bindings.bindings.preprocess_config;
    preprocess_config_explicit_ = bindings.bindings.preprocess_config_explicit;

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

    if (!upstream_camera_) {
        bindUpstreamCamera();
    }

    // Request full-resolution STREAM frames for preview (used as WebSocket preview source)
    if (upstream_camera_ && config_.websocket) {
        upstream_camera_->set_deliver_stream_frame(true);
        std::cout << "[ModelNode] stream_frame delivery enabled on camera " << camera_id << std::endl;
    } else if (!upstream_camera_) {
        std::cout << "[ModelNode] stream_frame delivery SKIPPED: no upstream camera found" << std::endl;
    } else {
        std::cout << "[ModelNode] stream_frame delivery SKIPPED: websocket disabled" << std::endl;
    }

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
        uint64_t sf_count = stream_frame_count_.load();
        uint64_t if_count = infer_frame_count_.load();
        uint64_t total = sf_count + if_count;
        response("get_stats", MA_OK, {
            {"infer_count", infer_count_.load()},
            {"error_count", error_count_.load()},
            {"infer_ema_ms", infer_ema_ms_},
            {"ws_event_count", ws_event_count_.load()},
            {"ws_clients", ws_ ? ws_->client_count() : 0},
            {"stream_frame_count", sf_count},
            {"infer_frame_fallback_count", if_count},
            {"stream_frame_ratio", total > 0 ? static_cast<double>(sf_count) / total : 0.0},
            {"stream_frame_active", sf_count > 0}
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
        // Prefer full-resolution STREAM frame for preview when available;
        // fall back to INFER frame for backward compatibility.
        const lua_cv::Frame& preview_frame =
            ctx.has_stream_frame() ? ctx.stream_frame->frame() : ctx.frame->frame();

        if (ctx.has_stream_frame()) {
            uint64_t cnt = stream_frame_count_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (cnt == 1) {
                std::cout << "[ModelNode] First stream_frame received (full-res preview active)" << std::endl;
            }
        } else {
            uint64_t cnt = infer_frame_count_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (cnt == 1) {
                std::cout << "[ModelNode] WARN: stream_frame not available, using infer frame for preview" << std::endl;
            }
        }

        ModelPreviewFormatConfig preview_config;
        preview_config.preview_width = config_.preview_width;
        preview_config.preview_height = config_.preview_height;
        preview_config.preview_fps = config_.preview_fps;

        nlohmann::json ws_msg = build_model_preview_message(
            event_data,
            preview_frame,
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

        // Lazy camera binding: camera may have been added to dependencies_ AFTER onStart()
        // (Node-RED creates model before camera). By the time we receive the first frame,
        // camera is guaranteed to be in dependencies_ (addDependency is called before
        // setupDataFlow which connected our inbox).
        if (!upstream_camera_) {
            bindUpstreamCamera();
            if (upstream_camera_ && config_.websocket) {
                upstream_camera_->set_deliver_stream_frame(true);
                std::cout << "[ModelNode] Lazily enabled stream_frame delivery on camera "
                          << upstream_camera_->id() << std::endl;
            }
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

            nlohmann::json event_data = build_inference_event_data(std::move(result));

            event("invoke", MA_OK, event_data);

            // Broadcast preview: maybeBroadcastPreview uses stream_frame (full-res) from
            // PipelineContext if available, otherwise falls back to infer frame.
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
    auto t_total_start = std::chrono::steady_clock::now();
    FullFrameExecutionResult execution = execute_full_frame_inference(
        frame,
        ModelExecutorConfig{
            session_.get(),
            &preprocess_config_,
            config_.crop_size_explicit,
            config_.crop_width,
            config_.crop_height,
        });
    if (execution.warning) {
        event("warning", 0, {
            {"message", execution.warning->message},
            {"detail", execution.warning->detail}
        });
    }

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
        execution.preprocess_meta);

    auto t_post_start = std::chrono::steady_clock::now();
    nlohmann::json result = callPostprocess(
        std::move(execution.outputs), std::move(execution.output_shapes), meta);
    auto t_post_end = std::chrono::steady_clock::now();
    execution.timings.postprocess_ms = elapsed_ms(t_post_start, t_post_end);
    if (config_.profile) {
        emitProfile(build_full_frame_profile_payload(
            execution.timings,
            execution.preprocess_meta,
            elapsed_ms(t_total_start, t_post_end)));
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
    double postprocess_ms_local = 0.0;
    RoiExecutionResult execution = execute_roi_inference(
        frame,
        roi,
        ModelExecutorConfig{
            session_.get(),
            &preprocess_config_,
            config_.crop_size_explicit,
            config_.crop_width,
            config_.crop_height,
        });
    if (execution.warning) {
        event("warning", 0, {
            {"message", execution.warning->message},
            {"detail", execution.warning->detail}
        });
    }
    if (!execution.valid) {
        return nullptr;
    }

    nlohmann::json meta = build_roi_meta(roi, upstream, config_.conf_threshold, execution.preprocess_meta);
    auto t_post_start = std::chrono::steady_clock::now();
    auto item = callPostprocess(
        std::move(execution.outputs), std::move(execution.output_shapes), meta);
    auto t_post_end = std::chrono::steady_clock::now();
    postprocess_ms_local = elapsed_ms(t_post_start, t_post_end);

    if (metrics) {
        metrics->preprocess_total_ms += execution.preprocess_ms;
        metrics->postprocess_total_ms += postprocess_ms_local;
        metrics->vpss_total_ms += execution.vpss_ms;
        metrics->cpu_pre_total_ms += execution.cpu_pre_ms;
        metrics->build_input_total_ms += execution.build_input_ms;
        metrics->infer_total_ms += execution.infer_ms;
        metrics->tpu_input_total_ms += execution.tpu_input_ms;
        metrics->tpu_forward_total_ms += execution.tpu_forward_ms;
        metrics->tpu_output_total_ms += execution.tpu_output_ms;
        if (execution.vpss_attempted) {
            metrics->vpss_attempted_count++;
        }
        if (execution.use_vb) {
            metrics->use_vb_count++;
        }
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
        ctx->frame->ref();
        if (ctx->stream_frame) ctx->stream_frame->ref();

        auto* next_ctx = new PipelineContext{
            ctx->frame,
            ctx->stream_frame,
            result,
            ctx->frame_id
        };

        if (!mbox->post(next_ctx, 0)) {
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
