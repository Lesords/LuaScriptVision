#include "camera_node.h"
#include "resource_estimator.h"

#include <algorithm>
#include <chrono>

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
            event("error", MA_EIO, {{"message", "Failed to open camera"}});
            return MA_EIO;
        }
    }
#endif

    running_.store(true, std::memory_order_release);
    capture_thread_ = std::thread(&CameraNode::captureLoop, this);

#ifdef USE_CVI_CAMERA
    // sscma-node compatibility: Start internal VENC
    if (config_.websocket && config_.stream_to_frontend) {
        if (!initStreamEncoder()) {
            event("error", MA_EIO, {{"message", "Failed to init stream encoder"}});
        } else {
            // Send event to notify frontend WebSocket is ready
            event("websocket", MA_OK, {
                {"port", config_.ws_port},
                {"codec", "h264"},
                {"type", "video"}
            });
        }
    }
#endif

    // Notify Node-RED that camera is active
    event("enabled", MA_OK, {{"value", true}});

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
        if (!data.contains("value")) {
            return MA_EINVAL;
        }
        config_.fps = data.at("value").get<double>();
        return MA_OK;
    }

    if (action == "set_infer_fps_limit") {
        if (!data.contains("value")) {
            return MA_EINVAL;
        }
        config_.infer_fps_limit = data.at("value").get<double>();
        ResourceEstimator::instance().register_camera(
            id_, frame_skip_enabled(), infer_fps_limit());
        return MA_OK;
    }

    if (action == "enabled") {
        bool enabled = data.value("value", true);
        inference_enabled_.store(enabled, std::memory_order_release);
        event("enabled", MA_OK, {{"value", enabled}});
        return MA_OK;
    }

    // Silently accept unsupported camera controls (light, pause, etc.)
    // Return MA_OK so Node-RED does not show an error status
    if (action == "light" || action == "pause") {
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

void CameraNode::captureLoop() {
    while (running_.load(std::memory_order_acquire)) {
#ifdef USE_CVI_CAMERA
        lua_cv::Frame frame;

        bool log_error = (skip_state_.camera_nobuf_streak == 0);
        if (!camera_ || !camera_->read(frame, 100, log_error)) {
            nobuf_count_.fetch_add(1, std::memory_order_relaxed);

            // Handle NOBUF backoff
            skip_state_.camera_nobuf_streak++;
            if (skip_state_.camera_nobuf_streak >= computeNobufThreshold()) {
                applyExponentialBackoff();
                if (!running_.load(std::memory_order_acquire)) {
                    break;
                }
                auto now = std::chrono::steady_clock::now();
                if (skip_state_.next_infer_time > now) {
                    auto cooldown = skip_state_.next_infer_time - now;
                    if (cooldown > std::chrono::milliseconds(1)) {
                        std::this_thread::sleep_for(cooldown);
                    }
                }
            }
            continue;
        }

        // Reset NOBUF streak on successful read
        skip_state_.camera_nobuf_streak = 0;
        skip_state_.backoff_exponent = 0;

        // Check if we should skip this frame
        if (shouldSkipFrame()) {
            skip_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Process and distribute frame
        if (processFrame(frame)) {
            frame_count_.fetch_add(1, std::memory_order_relaxed);
        }
#else
        // Simulation mode without CVI Camera
        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 FPS
#endif
    }
}

bool CameraNode::processFrame(lua_cv::Frame& frame) {
    uint64_t frame_id = next_frame_id_++;

    // Create SharedFrame (reference counting wrapper)
    auto* sf = new SharedFrame(std::move(frame));
    sf->set_timestamp(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()));
    sf->set_frame_id(frame_id);

    // Distribute to inference channel subscribers
    if (config_.enable_inference && inference_enabled_.load(std::memory_order_acquire)) {
        distributeFrame(sf, frame_id, FrameChannel::INFER);
    }

    // Note: Stream channel uses hardware binding (VPSS->VENC), not MessageBox

    // Release initial reference (subscribers hold subsequent references)
    sf->release();

    // Update skip timing based on downstream feedback
    updateSkipTiming();

    return true;
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

void CameraNode::distributeFrame(SharedFrame* sf, uint64_t frame_id, FrameChannel channel) {
    std::lock_guard<std::mutex> lock(channel_mutex_);
    auto& subs = channel_subscribers_[static_cast<size_t>(channel)];

    for (auto* mbox : subs) {
        // Increase reference count before creating PipelineContext
        sf->ref();

        auto* ctx = new PipelineContext{
            sf,
            nlohmann::json{},  // Empty upstream_result (CameraNode is source)
            frame_id
        };

        if (!mbox->post(ctx, 0)) {
            // Post failed, cleanup (delete calls destructor which calls release)
            delete ctx;
        }
    }
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
    if (!camera_ || !camera_->is_opened()) {
        return false;
    }

    // Configure VENC with channel 2 (sscma-node compatible CHN_H264)
    lua_cv::VencEncoder::Config enc_cfg;
    enc_cfg.channel = static_cast<VENC_CHN>(config_.venc_channel);  // Default 2
    enc_cfg.width = static_cast<uint32_t>(config_.width);
    enc_cfg.height = static_cast<uint32_t>(config_.height);
    enc_cfg.fps = static_cast<uint32_t>(config_.fps);
    enc_cfg.bitrate_kbps = static_cast<uint32_t>(config_.bitrate_kbps);
    enc_cfg.codec = lua_cv::VencEncoder::CodecType::H264;
    enc_cfg.gop = 30;  // 1 second between keyframes

    // Create VENC
    stream_encoder_.encoder = std::make_unique<lua_cv::VencEncoder>(enc_cfg);
    if (!stream_encoder_.encoder->init()) {
        return false;
    }

    // Bind to VPSS Stream Channel
    int vpss_grp = camera_->vpss_group();
    int vpss_chn = camera_->vpss_stream_channel();
    if (!stream_encoder_.encoder->bind_to_vpss(
            static_cast<VPSS_GRP>(vpss_grp),
            static_cast<VPSS_CHN>(vpss_chn))) {
        stream_encoder_.encoder->shutdown();
        return false;
    }

    // Create WebSocket
    lua_cv::WebSocketTransport::Config ws_cfg;
    ws_cfg.port = config_.ws_port;
    ws_cfg.path = "/";  // Root path
    stream_encoder_.ws = std::make_unique<lua_cv::WebSocketTransport>(ws_cfg);
    if (!stream_encoder_.ws->start()) {
        stream_encoder_.encoder->shutdown();
        return false;
    }

    // Request IDR when a new browser client connects so it gets a full I-frame.
    auto* enc_ptr = stream_encoder_.encoder.get();
    stream_encoder_.ws->set_new_client_callback([enc_ptr]() {
        if (enc_ptr) {
            enc_ptr->request_idr();
        }
    });

    // Start encode thread
    stream_encoder_.running.store(true, std::memory_order_release);
    stream_encoder_.encode_thread = std::thread(&CameraNode::streamEncodeLoop, this);

    return true;
}

void CameraNode::streamEncodeLoop() {
    auto& encoder = stream_encoder_.encoder;
    auto& ws = stream_encoder_.ws;
    auto& running = stream_encoder_.running;

    uint64_t frame_count = 0;
    running.store(true, std::memory_order_release);

    while (running.load(std::memory_order_acquire)) {
        if (!encoder) break;

        // Get encoded stream
        lua_cv::VencEncoder::EncodedStream stream;
        if (!encoder->get_stream(&stream, 1000)) {
            continue;
        }

        // Broadcast via WebSocket (pure H.264 binary)
        if (ws) {
            ws->broadcast_binary(stream.data.data(), stream.data.size(), stream.is_keyframe);
            frame_count++;
        }

        encoder->release_stream();
    }
}
#endif

} // namespace node
