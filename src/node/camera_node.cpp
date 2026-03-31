#include "camera_node.h"
#include "resource_estimator.h"

#include <algorithm>
#include <chrono>
#include <iostream>

// Include CV modules when available
#ifdef USE_CVI_CAMERA
#include "modules/cv/mmf_context.h"
#include "modules/cv/cvi_camera.h"
#include "stream/venc_encoder.h"
#include "stream/websocket_transport.h"
#endif

namespace node {

// Register CameraNode as singleton type
REGISTER_NODE_SINGLETON("camera", CameraNode);

CameraNode::CameraNode(const std::string& id, const std::string& type)
    : DataNode(id, type, 4) {}

CameraNode::~CameraNode() {
    onDestroy();
}

int CameraNode::onCreate(const nlohmann::json& config) {
    // Parse configuration
    if (config.contains("width")) {
        config_.width = config["width"].get<int>();
    }
    if (config.contains("height")) {
        config_.height = config["height"].get<int>();
    }
    if (config.contains("fps")) {
        config_.fps = config["fps"].get<double>();
    }
    if (config.contains("infer_fps_limit")) {
        config_.infer_fps_limit = config["infer_fps_limit"].get<double>();
    }
    if (config.contains("sensor")) {
        config_.sensor = config["sensor"].get<std::string>();
    }
    if (config.contains("enable_stream")) {
        config_.enable_stream = config["enable_stream"].get<bool>();
    }
    if (config.contains("enable_inference")) {
        config_.enable_inference = config["enable_inference"].get<bool>();
    }
    if (config.contains("preview")) {
        config_.preview = config["preview"].get<bool>();
    }
    if (config.contains("light")) {
        config_.light = config["light"].get<int>();
    }
    if (config.contains("deliver_stream_frame")) {
        deliver_stream_frame_.store(config["deliver_stream_frame"].get<bool>(),
                                   std::memory_order_release);
    }

    std::cout << "[CameraNode] Configuration: " << config_.width << "x" << config_.height
              << " @ " << config_.fps << " fps, sensor=" << config_.sensor
              << ", enable_stream=" << config_.enable_stream
              << ", enable_inference=" << config_.enable_inference << std::endl;

    // sscma-node compatibility config
    if (config.contains("websocket")) {
        config_.websocket = config["websocket"].get<bool>();
    }
    if (config.contains("ws_port")) {
        config_.ws_port = config["ws_port"].get<int>();
    }
    if (config.contains("bitrate_kbps")) {
        config_.bitrate_kbps = config["bitrate_kbps"].get<int>();
    }
    if (config.contains("stream_to_frontend")) {
        config_.stream_to_frontend = config["stream_to_frontend"].get<bool>();
    }
    if (config.contains("venc_channel")) {
        config_.venc_channel = config["venc_channel"].get<int>();
    }

    if (config_.websocket) {
        std::cout << "[CameraNode] WebSocket enabled: port=" << config_.ws_port
                  << ", stream_to_frontend=" << config_.stream_to_frontend << std::endl;
    }

#ifdef USE_CVI_CAMERA
    // Setup VB pools
    if (!setupVbPools()) {
        event("error", MA_ENOMEM, {{"message", "Failed to setup VB pools"}});
        return MA_ENOMEM;
    }

    // Create camera instance
    try {
        camera_ = std::make_unique<lua_cv::CviCamera>();
    } catch (const std::exception& e) {
        event("error", MA_EIO, {
            {"message", "Failed to create camera"},
            {"detail", e.what()}
        });
        cleanupVbPools();
        return MA_EIO;
    }
#endif

    // Send create response with configuration
    response("create", MA_OK, {
        {"width", config_.width},
        {"height", config_.height},
        {"fps", static_cast<int>(config_.fps)}
    });

    return MA_OK;
}

int CameraNode::onStart() {
    if (running_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

#ifdef USE_CVI_CAMERA
    // Open camera
    if (camera_ && !camera_->is_opened()) {
        if (!camera_->open()) {
            std::cerr << "[CameraNode] Failed to open camera" << std::endl;
            event("error", MA_EIO, {{"message", "Failed to open camera"}});
            return MA_EIO;
        }
        std::cout << "[CameraNode] Camera opened successfully" << std::endl;
    }
#endif

    running_.store(true, std::memory_order_release);
    capture_thread_ = std::thread(&CameraNode::captureLoop, this);

#ifdef USE_CVI_CAMERA
    // sscma-node compatibility: Start internal VENC
    if (config_.websocket && config_.stream_to_frontend) {
#ifdef USE_CVI_MPI
        std::cout << "[CameraNode] Initializing stream encoder..." << std::endl;
        if (!initStreamEncoder()) {
            std::cerr << "[CameraNode] Failed to init stream encoder" << std::endl;
            event("error", MA_EIO, {{"message", "Failed to init stream encoder"}});
        } else {
            std::cout << "[CameraNode] Stream encoder initialized, sending websocket event" << std::endl;
            // Send event to notify frontend WebSocket is ready
            event("websocket", MA_OK, {
                {"port", config_.ws_port},
                {"codec", "h264"},
                {"type", "video"}
            });
        }
#else
        std::cerr << "[CameraNode] WebSocket streaming not available (USE_CVI_MPI not defined)" << std::endl;
        event("error", MA_EINVAL, {{"message", "WebSocket streaming requires USE_CVI_MPI"}});
#endif
    }
#endif

    // Send enabled event - data must be direct boolean for frontend comparison
    event("enabled", MA_OK, true);
    return MA_OK;
}

int CameraNode::onStop() {
#ifdef USE_CVI_CAMERA
    // sscma-node compatibility: Stop stream encoder
    if (stream_encoder_.running.load(std::memory_order_acquire)) {
        stream_encoder_.running.store(false, std::memory_order_release);
        if (stream_encoder_.encode_thread.joinable()) {
            stream_encoder_.encode_thread.join();
        }
    }

    // Cleanup WebSocket
    if (stream_encoder_.ws) {
        stream_encoder_.ws->stop();
        stream_encoder_.ws.reset();
    }

    // Cleanup VENC
    if (stream_encoder_.encoder) {
        stream_encoder_.encoder->shutdown();
        stream_encoder_.encoder.reset();
    }
#endif

    stopCapture();

#ifdef USE_CVI_CAMERA
    if (camera_) {
        camera_->release();
    }
#endif

    // Send enabled event - data must be direct boolean for frontend comparison
    event("enabled", MA_OK, false);

    return MA_OK;
}

void CameraNode::stopCapture() {
    bool was_running = running_.exchange(false, std::memory_order_acq_rel);
    if (was_running && capture_thread_.joinable()) {
        capture_thread_.join();
    }
}

int CameraNode::onDestroy() {
    onStop();

#ifdef USE_CVI_CAMERA
    camera_.reset();
    cleanupVbPools();
#endif

    return MA_OK;
}

int CameraNode::onControl(const std::string& action, const nlohmann::json& data) {
    if (action == "get_stats") {
        response("get_stats", MA_OK, {
            {"frame_count", frame_count_.load()},
            {"skip_count", skip_count_.load()},
            {"nobuf_count", nobuf_count_.load()},
            {"infer_ema_ms", skip_state_.infer_ema_ms}
        });
        return MA_OK;
    }

    if (action == "set_fps") {
        // Support both formats: {"value": N} or direct number
        if (data.contains("value")) {
            config_.fps = data.at("value").get<double>();
        } else if (data.is_number()) {
            config_.fps = data.get<double>();
        } else {
            return MA_EINVAL;
        }
        return MA_OK;
    }

    if (action == "set_infer_fps_limit") {
        // Support both formats: {"value": N} or direct number
        if (data.contains("value")) {
            config_.infer_fps_limit = data.at("value").get<double>();
        } else if (data.is_number()) {
            config_.infer_fps_limit = data.get<double>();
        } else {
            return MA_EINVAL;
        }
        ResourceEstimator::instance().register_camera(
            id_, frame_skip_enabled(), infer_fps_limit());
        return MA_OK;
    }

    if (action == "preview") {
        // Support both formats: {"value": bool} or direct boolean
        if (data.contains("value")) {
            config_.preview = data.at("value").get<bool>();
        } else if (data.is_boolean()) {
            config_.preview = data.get<bool>();
        } else {
            return MA_EINVAL;
        }
        response("preview", MA_OK, {{"preview", config_.preview}});
        return MA_OK;
    }

    if (action == "light") {
        // Support both formats: {"value": N} or direct number N
        if (data.contains("value")) {
            config_.light = data.at("value").get<int>();
        } else if (data.is_number()) {
            config_.light = data.get<int>();
        } else {
            return MA_EINVAL;
        }
        response("light", MA_OK, {{"light", config_.light}});
        return MA_OK;
    }

    if (action == "enabled") {
        bool enabled = data.value("value", true);
        inference_enabled_.store(enabled, std::memory_order_release);
        std::cout << "[CameraNode] Inference " << (enabled ? "enabled" : "disabled") << std::endl;
        // Send event - data must be direct boolean for frontend comparison
        event("enabled", MA_OK, enabled);
        return MA_OK;
    }

    return MA_EINVAL;
}

void CameraNode::attach(FrameChannel ch, ContextBox* subscriber) {
    if (!subscriber) return;
    std::lock_guard<std::mutex> lock(channel_mutex_);
    auto& subs = channel_subscribers_[static_cast<size_t>(ch)];
    subs.push_back(subscriber);
}

void CameraNode::detach(FrameChannel ch, ContextBox* subscriber) {
    if (!subscriber) return;
    std::lock_guard<std::mutex> lock(channel_mutex_);
    auto& subs = channel_subscribers_[static_cast<size_t>(ch)];
    subs.erase(std::remove(subs.begin(), subs.end(), subscriber), subs.end());
}

void CameraNode::report_proc_time(const std::string& node_id, double proc_ms) {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    auto now = std::chrono::steady_clock::now();
    downstream_timings_[node_id] = {proc_ms, now};
}

bool CameraNode::get_stream_binding(int* vpss_grp, int* vpss_chn) const {
    if (!vpss_grp || !vpss_chn) {
        return false;
    }
#ifdef USE_CVI_CAMERA
    if (!camera_) {
        return false;
    }
    *vpss_grp = camera_->vpss_group();
    *vpss_chn = camera_->vpss_stream_channel();
    return *vpss_grp >= 0 && *vpss_chn >= 0;
#else
    (void)vpss_grp;
    (void)vpss_chn;
    return false;
#endif
}

bool CameraNode::get_infer_binding(int* vpss_grp, int* vpss_chn) const {
    if (!vpss_grp || !vpss_chn) {
        return false;
    }
#ifdef USE_CVI_CAMERA
    if (!camera_) {
        return false;
    }
    *vpss_grp = camera_->vpss_group();
    *vpss_chn = camera_->vpss_infer_channel();
    return *vpss_grp >= 0 && *vpss_chn >= 0;
#else
    (void)vpss_grp;
    (void)vpss_chn;
    return false;
#endif
}

bool CameraNode::get_preview_binding(int* vpss_grp, int* vpss_chn) const {
    if (!vpss_grp || !vpss_chn) {
        return false;
    }
#ifdef USE_CVI_CAMERA
    if (!camera_ || !camera_->vpss_preview_enabled()) {
        return false;
    }
    *vpss_grp = camera_->vpss_group();
    *vpss_chn = camera_->vpss_preview_channel();
    return *vpss_grp >= 0 && *vpss_chn >= 0;
#else
    (void)vpss_grp;
    (void)vpss_chn;
    return false;
#endif
}

void CameraNode::captureLoop() {
    while (running_.load(std::memory_order_acquire)) {
#ifdef USE_CVI_CAMERA
        // === STREAM Channel: capture into shared memory (latest_stream_sf_) ===
        // grab_latest_stream_frame() allows any node to get the latest full-resolution
        // NV21 frame. Hardware JPEG preview (VPSS Chn2 → VENC) runs independently.
        if (camera_) {
            lua_cv::Frame stream_frame;
            if (captureStreamFrame(stream_frame)) {
                auto* new_sf = new SharedFrame(std::move(stream_frame));
                new_sf->set_timestamp(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count()));

                std::lock_guard<std::mutex> lock(latest_stream_mutex_);
                if (latest_stream_sf_) latest_stream_sf_->release();
                latest_stream_sf_ = new_sf;
            }
        }

        // === INFER Channel: capture and process (pure infer, no stream coupling) ===
        lua_cv::Frame infer_frame;
        if (captureInferFrame(infer_frame)) {
            processInferFrame(infer_frame);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
#else
        // Simulation mode without CVI Camera
        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 FPS
#endif
    }

    // Cleanup latest stream frame on exit
    std::lock_guard<std::mutex> lock(latest_stream_mutex_);
    if (latest_stream_sf_) {
        latest_stream_sf_->release();
        latest_stream_sf_ = nullptr;
    }
}

// ========================================
// INFER Channel: Capture
// ========================================
bool CameraNode::captureInferFrame(lua_cv::Frame& frame) {
#ifdef USE_CVI_CAMERA
    bool log_error = (skip_state_.camera_nobuf_streak == 0);
    if (!camera_ || !camera_->read(frame, 100, log_error)) {
        nobuf_count_.fetch_add(1, std::memory_order_relaxed);

        // Handle NOBUF backoff
        skip_state_.camera_nobuf_streak++;
        if (skip_state_.camera_nobuf_streak >= computeNobufThreshold()) {
            applyExponentialBackoff();
            if (!running_.load(std::memory_order_acquire)) {
                return false;
            }

            // Wait for cooldown time before returning
            // This prevents immediate retry and gives VPSS time to recover
            auto now = std::chrono::steady_clock::now();
            if (skip_state_.next_infer_time > now) {
                auto cooldown = std::chrono::duration_cast<std::chrono::milliseconds>(
                    skip_state_.next_infer_time - now);
                if (cooldown > std::chrono::milliseconds(1)) {
                    std::this_thread::sleep_for(cooldown);
                }
            }
        }
        return false;
    }

    // Reset NOBUF streak on successful read
    skip_state_.camera_nobuf_streak = 0;
    skip_state_.backoff_exponent = 0;
    return true;
#else
    (void)frame;
    return false;
#endif
}

// ========================================
// INFER Channel: Process and distribute
// ========================================
bool CameraNode::processInferFrame(lua_cv::Frame& infer_frame) {
    // Check if we should skip this frame (infer FPS limit)
    if (shouldSkipFrame()) {
        skip_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    uint64_t frame_id = next_frame_id_++;

    // Create SharedFrame for INFER channel
    auto* infer_sf = new SharedFrame(std::move(infer_frame));
    infer_sf->set_timestamp(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()));
    infer_sf->set_frame_id(frame_id);

    // Check if there are INFER subscribers
    if (!hasInferSubscribers()) {
        infer_sf->release();
        return false;
    }

    // INFER channel only: stream_frame is always nullptr here.
    // The STREAM frame is captured independently in captureLoop() and stored
    // in latest_stream_sf_. ModelNode retrieves it via grab_latest_stream_frame()
    // after inference completes, keeping the two channels fully decoupled.
    auto* ctx = new PipelineContext{
        infer_sf,         // INFER frame (scaled, for inference)
        nullptr,          // STREAM frame: delivered independently, not via PipelineContext
        nlohmann::json{}, // Empty upstream_result (from CameraNode)
        frame_id
    };

    // Distribute to INFER channel subscribers
    if (config_.enable_inference && inference_enabled_.load(std::memory_order_acquire)) {
        distributeWithContext(ctx);
    }

    // Cleanup: PipelineContext deletion handles reference counting
    // Note: distributeWithContext increments refs for each subscriber
    delete ctx;

    // Update skip timing based on downstream feedback
    updateSkipTiming();

    frame_count_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ========================================
// STREAM Channel: Capture
// ========================================
bool CameraNode::captureStreamFrame(lua_cv::Frame& frame) {
#ifdef USE_CVI_CAMERA
    if (!camera_) {
        return false;
    }

    // Use short timeout to avoid blocking the capture loop
    if (!camera_->read_stream(frame, 10)) {
        static std::atomic<uint32_t> stream_fail_count{0};
        uint32_t cnt = stream_fail_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (cnt == 1 || cnt % 300 == 0) {
            std::cerr << "[CameraNode] WARN: STREAM frame capture failed"
                      << " (count=" << cnt << ")" << std::endl;
        }
        return false;
    }
    return true;
#else
    (void)frame;
    return false;
#endif
}

// ========================================
// Helper methods
// ========================================
bool CameraNode::hasInferSubscribers() const {
    std::lock_guard<std::mutex> lock(channel_mutex_);
    return !channel_subscribers_[static_cast<size_t>(FrameChannel::INFER)].empty();
}

bool CameraNode::hasStreamSubscribers() const {
    std::lock_guard<std::mutex> lock(channel_mutex_);
    return !channel_subscribers_[static_cast<size_t>(FrameChannel::STREAM)].empty();
}

bool CameraNode::shouldSkipFrame() {
    auto now = std::chrono::steady_clock::now();

    // Time window check
    if (now < skip_state_.next_infer_time) {
        return true;  // Skip
    }

    return false;  // Don't skip
}

void CameraNode::updateSkipTiming() {
    // Get maximum downstream processing time
    double max_proc_ms = getMaxDownstreamProcMs();

    // EWMA smoothing
    if (max_proc_ms > 0) {
        skip_state_.infer_ema_ms =
            skip_state_.kEmaAlpha * max_proc_ms +
            (1.0 - skip_state_.kEmaAlpha) * skip_state_.infer_ema_ms;
    }

    // Calculate next inference time
    double frame_interval_ms = 1000.0 / effectiveInferFps();
    double skip_interval_ms = std::max(
        skip_state_.infer_ema_ms * skip_state_.kSafetyFactor,
        frame_interval_ms
    );

    skip_state_.next_infer_time = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(skip_interval_ms)
        );

    skip_state_.skip_state_ready = true;
}

double CameraNode::getMaxDownstreamProcMs() const {
    std::lock_guard<std::mutex> lock(timing_mutex_);

    auto now = std::chrono::steady_clock::now();
    double max_proc_ms = 0.0;

    // Find max (don't erase in const method - stale entries handled elsewhere)
    for (const auto& [node_id, timing] : downstream_timings_) {
        if (now - timing.last_report <= kTimingStaleThreshold) {
            max_proc_ms = std::max(max_proc_ms, timing.proc_ms);
        }
    }

    return max_proc_ms;
}

void CameraNode::distributeWithContext(PipelineContext* ctx) {
    if (!ctx) return;

    std::lock_guard<std::mutex> lock(channel_mutex_);

    // Determine target channel based on which frame is present
    FrameChannel target_channel;
    if (ctx->frame != nullptr) {
        target_channel = FrameChannel::INFER;
    } else if (ctx->stream_frame != nullptr) {
        target_channel = FrameChannel::STREAM;
    } else {
        return;  // No frames to distribute
    }

    auto& subs = channel_subscribers_[static_cast<size_t>(target_channel)];

    for (auto* mbox : subs) {
        // Create a new PipelineContext for each subscriber
        auto* ctx_copy = new PipelineContext{
            ctx->frame,
            ctx->stream_frame,
            ctx->upstream_result,
            ctx->frame_id
        };

        // Increment reference counts for this subscriber
        if (ctx_copy->frame) {
            ctx_copy->frame->ref();
        }
        if (ctx_copy->stream_frame) {
            ctx_copy->stream_frame->ref();
        }

        if (!mbox->post(ctx_copy, 0)) {
            // Post failed, cleanup
            delete ctx_copy;
        }
    }
}

SharedFrame* CameraNode::grab_latest_stream_frame() {
    std::lock_guard<std::mutex> lock(latest_stream_mutex_);
    if (!latest_stream_sf_) return nullptr;
    latest_stream_sf_->ref();  // caller owns this reference; must call release()
    return latest_stream_sf_;
}

int CameraNode::computeNobufThreshold() const {
    // Dynamic threshold based on downstream load
    double ratio = skip_state_.infer_ema_ms / (1000.0 / effectiveInferFps());
    if (ratio >= 2.0) return 1;
    if (ratio >= 1.3) return 2;
    return 3;
}

void CameraNode::applyExponentialBackoff() {
    // Exponential backoff: 1x -> 2x -> 4x -> 8x -> 16x (max 1000ms)
    int cooldown_base_ms = static_cast<int>(1000.0 / effectiveInferFps());
    int multiplier = 1 << skip_state_.backoff_exponent;
    int cooldown_ms = std::min(cooldown_base_ms * multiplier, 1000);

    skip_state_.next_infer_time = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(cooldown_ms);

    if (skip_state_.backoff_exponent < skip_state_.kMaxBackoffExponent) {
        skip_state_.backoff_exponent++;
    }
}

double CameraNode::effectiveInferFps() const {
    if (config_.infer_fps_limit > 0.0) {
        return std::min(config_.infer_fps_limit, config_.fps);
    }
    return config_.fps;
}

bool CameraNode::setupVbPools() {
#ifdef USE_CVI_CAMERA
    try {
        // MmfContext is a singleton - use instance() and init()
        lua_cv::MmfContext::Config config;
        if (!lua_cv::MmfContext::build_default_config(&config)) {
            return false;
        }
        return lua_cv::MmfContext::instance().init(config);
    } catch (...) {
        return false;
    }
#else
    return true;
#endif
}

void CameraNode::cleanupVbPools() {
#ifdef USE_CVI_CAMERA
    lua_cv::MmfContext::instance().shutdown();
#endif
}

#ifdef USE_CVI_CAMERA
bool CameraNode::initStreamEncoder() {
    // Create VENC encoder
    lua_cv::VencEncoder::Config venc_config;
    venc_config.codec = lua_cv::VencEncoder::CodecType::H264;
    venc_config.width = config_.width;
    venc_config.height = config_.height;
    venc_config.fps = static_cast<uint32_t>(config_.fps);
    venc_config.bitrate_kbps = config_.bitrate_kbps;
    venc_config.gop = venc_config.fps;  // 1 second I-frame interval
    venc_config.channel = static_cast<VENC_CHN>(config_.venc_channel);

    stream_encoder_.encoder = std::make_unique<lua_cv::VencEncoder>(venc_config);
    if (!stream_encoder_.encoder->init()) {
        std::cerr << "[CameraNode] VENC encoder init failed" << std::endl;
        return false;
    }

    // Bind VENC to VPSS stream channel
    int vpss_grp = camera_->vpss_group();
    int vpss_chn = camera_->vpss_stream_channel();
    if (vpss_grp < 0 || vpss_chn < 0) {
        std::cerr << "[CameraNode] Invalid VPSS binding for stream" << std::endl;
        return false;
    }

    if (!stream_encoder_.encoder->bind_to_vpss(
            static_cast<VPSS_GRP>(vpss_grp),
            static_cast<VPSS_CHN>(vpss_chn))) {
        std::cerr << "[CameraNode] Failed to bind VENC to VPSS" << std::endl;
        return false;
    }

    std::cout << "[CameraNode] VENC bound to VPSS grp=" << vpss_grp
              << " chn=" << vpss_chn << std::endl;

    // Create WebSocket transport
    lua_cv::WebSocketTransport::Config ws_config;
    ws_config.port = config_.ws_port;
    ws_config.max_clients = 8;

    stream_encoder_.ws = std::make_unique<lua_cv::WebSocketTransport>(ws_config);

    if (!stream_encoder_.ws->start()) {
        std::cerr << "[CameraNode] WebSocket start failed" << std::endl;
        return false;
    }

    std::cout << "[CameraNode] WebSocket started on port " << config_.ws_port << std::endl;

    // Start encode thread
    stream_encoder_.running.store(true, std::memory_order_release);
    stream_encoder_.encode_thread = std::thread(&CameraNode::streamEncodeLoop, this);

    return true;
}

void CameraNode::streamEncodeLoop() {
    while (stream_encoder_.running.load(std::memory_order_acquire)) {
        lua_cv::VencEncoder::EncodedStream stream;
        if (!stream_encoder_.encoder->get_stream(&stream, 100)) {
            continue;
        }

        // Broadcast to WebSocket clients
        if (stream_encoder_.ws && stream_encoder_.ws->is_running()) {
            stream_encoder_.ws->broadcast_binary(
                stream.data.data(),
                stream.data.size());
        }

        stream_encoder_.encoder->release_stream();
    }
}
#endif
} // namespace node
