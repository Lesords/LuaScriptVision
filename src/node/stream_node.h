#pragma once

#include "node.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace lua_cv {
class VencEncoder;
class RtspServer;
class WebSocketTransport;
}

namespace node {

class CameraNode;

class StreamNode : public Node {
public:
    StreamNode(const std::string& id, const std::string& type);
    ~StreamNode() override;

    int onCreate(const nlohmann::json& config) override;
    int onStart() override;
    int onStop() override;
    int onDestroy() override;
    int onControl(const std::string& action, const nlohmann::json& data) override;

private:
    struct Config {
        std::string codec = "h264";
        int bitrate_kbps = 2000;
        int fps = 30;
        int width = 1920;
        int height = 1080;
        int gop = 30;
        int ip_qp_delta = -1;
        int port = 554;
        std::string session = "live";
        std::string username = "admin";  // Node-RED compatibility
        std::string password = "admin";  // Node-RED compatibility
        bool websocket = false;
        int ws_port = 8080;
        std::string ws_path = "/";
        int ws_max_clients = 8;
    } config_;

    CameraNode* camera_node_ = nullptr;
    std::unique_ptr<lua_cv::VencEncoder> encoder_;
    std::unique_ptr<lua_cv::RtspServer> rtsp_;
    std::unique_ptr<lua_cv::WebSocketTransport> ws_;
    std::thread encode_thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> stream_frames_{0};
    std::atomic<uint64_t> ws_frames_{0};

    void encodeLoop();
    bool resolveStreamChannel(int* vpss_grp, int* vpss_chn) const;
};

} // namespace node
