#pragma once

#include "data_node.h"
#include "model_preprocess_utils.h"
#include "model_roi_utils.h"
#include "preprocess_config.h"
#include "luaref_json.h"

#ifdef USE_CVI_MPI
#include "modules/cv/cvi_vpss_processor.h"
#endif

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
                                         const SelectedRoi& roi,
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
    void previewLoop();
    void maybeBroadcastPreview(const lua_cv::Frame& frame, const nlohmann::json& event_data);

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
        std::vector<std::string> classes; // Enforced classes list

        // Lua integration
        bool profile = false;
        bool output = false;
        bool debug = false;

        // WebSocket configuration
        bool websocket = true;
        int ws_port = 8090;
        std::string ws_path = "/";
        int ws_max_clients = 8;

        // Preview video stream configuration (JPEG encoding via stream_frame from inbox)
        std::string preview_resolution = "";  // e.g., "640x640", "" = use original frame size
        int preview_fps = 15;       // Preview frame rate (default 15fps)

        // Parsed resolution values
        int preview_width = 0;      // Parsed from preview_resolution
        int preview_height = 0;     // Parsed from preview_resolution
    } config_;

    // Lua integration (independent State for thread safety)
    lua_State* L_ = nullptr;
    LuaIntf::LuaRef postprocess_;
    LuaIntf::LuaRef select_rois_;
    LuaIntf::LuaRef preprocess_config_ref_;
    PreprocessConfig preprocess_config_;
    bool preprocess_config_explicit_ = false;

    // TPU Session (shared via SessionManager)
    std::shared_ptr<inference::CviSession> session_;

    // Inference thread
    std::thread infer_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> infer_enabled_{true};
    std::unique_ptr<lua_cv::WebSocketTransport> ws_;

    // Preview thread: gets JPEG bytes from CameraNode, overlays inference results,
    // and broadcasts base64 JPEG over WebSocket.
#ifdef USE_CVI_MPI
    // Persistent VPSS preprocessor for inference: reused across frames to avoid per-frame
    // CVI_VPSS_CreateGrp/DestroyGrp which accumulates ION work buffers and causes OOM.
    std::unique_ptr<lua_cv::CviVpssProcessor> vpss_processor_;
#endif
    std::thread preview_thread_;
    std::atomic<bool> preview_running_{false};
    uint64_t last_preview_generation_ = 0;  // tracks last processed JPEG generation

    // Latest inference result (mutex-protected): inferLoop writes, previewLoop reads.
    mutable std::mutex infer_result_mutex_;
    nlohmann::json latest_infer_result_;

    // Upstream camera reference (for timing feedback and preview binding)
    CameraNode* upstream_camera_ = nullptr;

    // Statistics
    std::atomic<uint64_t> infer_count_{0};
    std::atomic<uint64_t> error_count_{0};
    std::atomic<uint64_t> ws_event_count_{0};
    std::atomic<uint64_t> stream_frame_count_{0};  // Preview frames successfully sent
    std::chrono::steady_clock::time_point last_preview_time_{};
    int preview_interval_ms_ = 66;  // ~15fps default (1000/15)
    double infer_ema_ms_ = 0.0;
    static constexpr double kEmaAlpha = 0.2;
};

} // namespace node
