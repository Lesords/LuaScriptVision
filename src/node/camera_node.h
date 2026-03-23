#pragma once

#include "data_node.h"
#include "node_factory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lua_cv {
class CviCamera;
class VencEncoder;
class WebSocketTransport;
}

namespace node {

// Frame distribution target channel
enum class FrameChannel {
    STREAM = 0,   // Video stream output (VPSS Chn0 -> VENC)
    INFER  = 1,   // Inference input (VPSS Chn1 -> ModelNode)
    SAVE   = 2,   // Save/snapshot (optional)
    COUNT
};

class CameraNode : public DataNode {
public:
    CameraNode(const std::string& id, const std::string& type);
    ~CameraNode() override;

    // Node lifecycle
    int onCreate(const nlohmann::json& config) override;
    int onStart() override;
    int onStop() override;
    int onDestroy() override;
    int onControl(const std::string& action, const nlohmann::json& data) override;

    // Channel-based subscription (overrides DataNode::attach/detach)
    void attach(FrameChannel ch, ContextBox* subscriber);
    void detach(FrameChannel ch, ContextBox* subscriber);

    // Downstream timing feedback (called by ModelNode)
    void report_proc_time(const std::string& node_id, double proc_ms);
    void stopCapture();

    // Statistics
    uint64_t frame_count() const { return frame_count_.load(std::memory_order_relaxed); }
    uint64_t skip_count() const { return skip_count_.load(std::memory_order_relaxed); }
    uint64_t nobuf_count() const { return nobuf_count_.load(std::memory_order_relaxed); }
    bool frame_skip_enabled() const { return config_.infer_fps_limit > 0.0; }
    double infer_fps_limit() const { return config_.infer_fps_limit; }
    bool get_stream_binding(int* vpss_grp, int* vpss_chn) const;
    bool get_infer_binding(int* vpss_grp, int* vpss_chn) const;

    // Configuration access (for ModelNode coordinate mapping)
    int config_width() const { return config_.width; }
    int config_height() const { return config_.height; }

private:
    // Configuration
    struct Config {
        int width = 1920;
        int height = 1080;
        double fps = 30.0;
        double infer_fps_limit = 0.0;
        std::string sensor = "ov5647";
        bool enable_stream = true;
        bool enable_inference = true;
        bool preview = false;
        int light = 0;
        // sscma-node compatibility config
        bool websocket = true;
        int ws_port = 8080;
        int bitrate_kbps = 4000;
        bool stream_to_frontend = true;
        int venc_channel = 2;
    } config_;

    // CV module components (reuse existing infrastructure)
    // Note: MmfContext is a singleton, accessed via MmfContext::instance()
#ifdef USE_CVI_CAMERA
    std::unique_ptr<lua_cv::CviCamera> camera_;

    // Stream encoder (sscma-node compatibility)
    struct StreamEncoder {
        std::atomic<bool> running{false};
        std::unique_ptr<lua_cv::VencEncoder> encoder;
        std::unique_ptr<lua_cv::WebSocketTransport> ws;
        std::thread encode_thread;
    } stream_encoder_;
#endif

    // Capture thread
    std::thread capture_thread_;
    std::atomic<bool> running_{false};
    uint64_t next_frame_id_ = 0;

    // Frame distribution (separated by channel)
    std::array<std::vector<ContextBox*>,
               static_cast<size_t>(FrameChannel::COUNT)> channel_subscribers_;
    mutable std::mutex channel_mutex_;

    // Adaptive frame skip state
    struct AdaptiveSkipState {
        double infer_ema_ms = 0.0;
        std::chrono::steady_clock::time_point next_infer_time{};
        bool skip_state_ready = false;
        int camera_nobuf_streak = 0;
        int backoff_exponent = 0;
        static constexpr double kEmaAlpha = 0.2;
        static constexpr double kSafetyFactor = 1.2;
        static constexpr int kMaxBackoffExponent = 4;  // max 16x cooldown
    } skip_state_;

    // Downstream timing feedback (parallel mode: multiple ModelNodes report)
    struct DownstreamTiming {
        double proc_ms = 0.0;
        std::chrono::steady_clock::time_point last_report{};
    };
    std::map<std::string, DownstreamTiming> downstream_timings_;
    mutable std::mutex timing_mutex_;
    static constexpr auto kTimingStaleThreshold = std::chrono::seconds(5);

    // Enable/disable inference distribution (controlled by "enabled" action)
    std::atomic<bool> inference_enabled_{true};

    // Statistics
    std::atomic<uint64_t> frame_count_{0};
    std::atomic<uint64_t> skip_count_{0};
    std::atomic<uint64_t> nobuf_count_{0};

    // Internal methods
    void captureLoop();
    bool processFrame(lua_cv::Frame& frame);
    bool shouldSkipFrame();
    void updateSkipTiming();
    double getMaxDownstreamProcMs() const;
    double effectiveInferFps() const;
    void distributeFrame(SharedFrame* sf, uint64_t frame_id, FrameChannel channel);
    int computeNobufThreshold() const;
    void applyExponentialBackoff();

    // VB pool management (reuse MMFContext)
    bool setupVbPools();
    void cleanupVbPools();

#ifdef USE_CVI_CAMERA
    // Stream encoder methods (sscma-node compatibility)
    bool initStreamEncoder();
    void streamEncodeLoop();
#endif
};

} // namespace node
