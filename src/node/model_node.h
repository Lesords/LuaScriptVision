#pragma once

#include "data_node.h"
#include "model_preprocess_utils.h"
#include "model_roi_utils.h"
#include "preprocess_config.h"
#include "luaref_json.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "LuaIntf.h"
#include <nlohmann/json.hpp>

// Forward declarations
namespace inference { class CviSession; }
namespace lua_cv {
class WebSocketTransport;
#ifdef USE_CVI_MPI
class VencEncoder;
#endif
}

namespace node {

class CameraNode;
struct RoiBatchMetrics;

// ModelNode: TPU inference with Lua postprocessing
class ModelNode : public DataNode {
public:
    enum InputMode {
        FULL_FRAME,      // Full frame + upstream metadata
        CROPPED_ROI      // VPSS crop from upstream boxes
    };

    ModelNode(const std::string& id, const std::string& type);
    ~ModelNode() override;

    // Node lifecycle
    int onCreate(const nlohmann::json& config) override;
    int onStart() override;
    int onStop() override;
    int onDestroy() override;
    int onControl(const std::string& action, const nlohmann::json& data) override;

    // Statistics
    uint64_t infer_count() const { return infer_count_.load(std::memory_order_relaxed); }
    uint64_t error_count() const { return error_count_.load(std::memory_order_relaxed); }
    double infer_ema_ms() const { return infer_ema_ms_; }

private:
    int parseConfig(const nlohmann::json& config);
    void parsePreviewConfig(const nlohmann::json& config);
    int initializeSession();
    int validateInputModeDependencies();
    void bindUpstreamCamera();

    void inferLoop();
    void maybeBroadcastPreview(const PipelineContext& ctx, const nlohmann::json& event_data);
    nlohmann::json runInference(const lua_cv::Frame& frame, const nlohmann::json& upstream);
    nlohmann::json runFullFrameInference(const lua_cv::Frame& frame, const nlohmann::json& upstream);
    nlohmann::json runCroppedRoiInference(const lua_cv::Frame& frame, const nlohmann::json& upstream);
    nlohmann::json runSingleRoiInference(const lua_cv::Frame& frame,
                                         const Roi& roi,
                                         const nlohmann::json& upstream,
                                         RoiBatchMetrics* metrics);
    nlohmann::json callPostprocess(
        std::vector<std::vector<float>> outputs,
        std::vector<std::vector<int64_t>> output_shapes,
        const nlohmann::json& meta);
    void forwardToDownstream(PipelineContext* ctx, const nlohmann::json& result);
    void updateEwma(double elapsed_ms);
    void cleanupLuaRef();
    void emitProfile(const nlohmann::json& profile);

#ifdef USE_CVI_MPI
    void h264EncodeLoop();
#endif

    // Configuration struct
    struct Config {
        // Model configuration
        std::string model_path;
        std::string script_path;
        InputMode input_mode = FULL_FRAME;
        int crop_width = 112;
        int crop_height = 112;
        bool crop_size_explicit = false;
        float conf_threshold = 0.25f;
        int infer_timeout_ms = 5000;

        // Lua integration
        bool profile = false;
        bool output = false;
        bool debug = false;

        // WebSocket configuration
        bool websocket = true;
        int ws_port = 8090;
        std::string ws_path = "/";
        int ws_max_clients = 8;

        // Preview video stream configuration (software JPEG encoding)
        std::string preview_resolution = "";  // e.g., "640x640", "" = use original frame size
        int preview_fps = 15;       // JPEG preview frame rate (default 15fps)

        // Parsed resolution values
        int preview_width = 0;      // Parsed from preview_resolution
        int preview_height = 0;     // Parsed from preview_resolution

        // H.264 hardware encoding configuration (default enabled)
        bool h264_preview = true;          // Enable H.264 hardware encoding (default: true)
        int h264_bitrate_kbps = 2000;      // H.264 bitrate
        int h264_gop = 30;                 // H.264 GOP size
    } config_;

    // Lua integration (independent State for thread safety)
    lua_State* L_ = nullptr;
    LuaIntf::LuaRef postprocess_;
    LuaIntf::LuaRef select_rois_;
    LuaIntf::LuaRef preprocess_config_ref_;
    PreprocessConfig preprocess_config_;
    bool preprocess_config_explicit_ = false;

    // TPU Session
    std::unique_ptr<inference::CviSession> session_;

    // Inference thread
    std::thread infer_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> infer_enabled_{true};  // Inference enable/disable control
    std::unique_ptr<lua_cv::WebSocketTransport> ws_;

#ifdef USE_CVI_MPI
    // H.264 hardware encoder (for video preview)
    std::unique_ptr<lua_cv::VencEncoder> encoder_;
    std::thread h264_encode_thread_;
    std::atomic<bool> h264_running_{false};
    bool h264_bound_to_vpss_ = false;
    int h264_bound_vpss_grp_ = -1;
    int h264_bound_vpss_chn_ = -1;
    std::atomic<bool> venc_initialized_{false};  // Track if VENC is properly initialized
    uint32_t venc_actual_width_ = 0;   // Actual VENC resolution (may differ from config)
    uint32_t venc_actual_height_ = 0;
#endif

    // Upstream camera reference (for timing feedback)
    CameraNode* upstream_camera_ = nullptr;

    // Statistics
    std::atomic<uint64_t> infer_count_{0};
    std::atomic<uint64_t> error_count_{0};
    std::atomic<uint64_t> ws_event_count_{0};
    std::chrono::steady_clock::time_point last_preview_time_{};  // For preview rate limiting
    int preview_interval_ms_;  // Calculated from preview_fps
    double infer_ema_ms_ = 0.0;
    static constexpr double kEmaAlpha = 0.2;
};

} // namespace node
