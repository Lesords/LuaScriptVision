#include "stream_node.h"
#include "camera_node.h"
#include "node_server.h"

#ifdef USE_CVI_MPI
#include "stream/venc_encoder.h"
#include "stream/rtsp_server.h"
#include "stream/websocket_transport.h"
#endif

#include <chrono>

namespace node {

REGISTER_NODE_SINGLETON("stream", StreamNode);

StreamNode::StreamNode(const std::string& id, const std::string& type)
    : Node(id, type) {
}

StreamNode::~StreamNode() {
    onDestroy();
}

int StreamNode::onCreate(const nlohmann::json& config) {
    if (config.contains("codec")) {
        config_.codec = config["codec"].get<std::string>();
    }
    if (config.contains("bitrate_kbps")) {
        config_.bitrate_kbps = config["bitrate_kbps"].get<int>();
    }
    if (config.contains("fps")) {
        config_.fps = config["fps"].get<int>();
    }
    if (config.contains("width")) {
        config_.width = config["width"].get<int>();
    }
    if (config.contains("height")) {
        config_.height = config["height"].get<int>();
    }
    if (config.contains("gop")) {
        config_.gop = config["gop"].get<int>();
    }
    if (config.contains("ip_qp_delta")) {
        config_.ip_qp_delta = config["ip_qp_delta"].get<int>();
    }
    if (config.contains("port")) {
        config_.port = config["port"].get<int>();
    }
    if (config.contains("session")) {
        config_.session = config["session"].get<std::string>();
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

    if (config_.fps <= 0 || config_.width <= 0 || config_.height <= 0) {
        event("error", MA_EINVAL, {{"message", "Invalid stream dimensions or fps"}});
        return MA_EINVAL;
    }
    if (config_.websocket && config_.codec != "h264") {
        event("error", MA_ENOTSUP, {{"message", "WebSocket video preview only supports H264 codec"}});
        return MA_ENOTSUP;
    }

    // Note: camera_node_ is resolved lazily in onStart() so that StreamNode can be
    // created before CameraNode (Node-RED may send create commands in any order).

    return MA_OK;
}

int StreamNode::onStart() {
    if (running_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

    // Resolve camera dependency here (lazy, supports any creation order)
    if (!camera_node_) {
        for (const auto& [dep_id, dep] : dependencies_) {
            if (dep && dep->type() == "camera") {
                camera_node_ = static_cast<CameraNode*>(dep);
                break;
            }
        }
    }

    if (!camera_node_ || !camera_node_->isStarted()) {
        return MA_EAGAIN;
    }

#ifdef USE_CVI_MPI
    int vpss_grp = -1;
    int vpss_chn = -1;
    if (!resolveStreamChannel(&vpss_grp, &vpss_chn)) {
        event("error", MA_EINVAL, {{"message", "Failed to resolve VPSS stream channel"}});
        return MA_EINVAL;
    }

    lua_cv::VencEncoder::Config enc_cfg;
    enc_cfg.width = static_cast<uint32_t>(config_.width);
    enc_cfg.height = static_cast<uint32_t>(config_.height);
    enc_cfg.fps = static_cast<uint32_t>(config_.fps);
    enc_cfg.bitrate_kbps = static_cast<uint32_t>(config_.bitrate_kbps);
    enc_cfg.gop = static_cast<uint32_t>(config_.gop);
    enc_cfg.ip_qp_delta = config_.ip_qp_delta;

    if (config_.codec == "h265") {
        enc_cfg.codec = lua_cv::VencEncoder::CodecType::H265;
    } else if (config_.codec == "jpeg") {
        enc_cfg.codec = lua_cv::VencEncoder::CodecType::JPEG;
    } else if (config_.codec == "mjpeg") {
        enc_cfg.codec = lua_cv::VencEncoder::CodecType::MJPEG;
    } else {
        enc_cfg.codec = lua_cv::VencEncoder::CodecType::H264;
    }

    encoder_ = std::make_unique<lua_cv::VencEncoder>(enc_cfg);
    if (!encoder_->init()) {
        event("error", MA_EIO, {{"message", "VENC encoder init failed"}});
        encoder_.reset();
        return MA_EIO;
    }

    if (!encoder_->bind_to_vpss(static_cast<VPSS_GRP>(vpss_grp),
                                static_cast<VPSS_CHN>(vpss_chn))) {
        event("error", MA_EIO, {{"message", "Failed to bind VPSS to VENC"}});
        encoder_->shutdown();
        encoder_.reset();
        return MA_EIO;
    }

    lua_cv::RtspServer::Config rtsp_cfg;
    rtsp_cfg.port = config_.port;
    rtsp_cfg.session_name = config_.session;
    rtsp_cfg.video_width = config_.width;
    rtsp_cfg.video_height = config_.height;
    rtsp_cfg.video_bitrate_kbps = config_.bitrate_kbps;

    if (config_.codec == "h265") {
        rtsp_cfg.video_codec = lua_cv::RtspServer::VideoCodec::H265;
    } else if (config_.codec == "jpeg" || config_.codec == "mjpeg") {
        rtsp_cfg.video_codec = lua_cv::RtspServer::VideoCodec::JPEG;
    } else {
        rtsp_cfg.video_codec = lua_cv::RtspServer::VideoCodec::H264;
    }

    rtsp_ = std::make_unique<lua_cv::RtspServer>(rtsp_cfg);
    if (!rtsp_->start()) {
        event("error", MA_EIO, {{"message", "RTSP server start failed"}});
        rtsp_.reset();
        encoder_->shutdown();
        encoder_.reset();
        return MA_EIO;
    }

    if (config_.websocket) {
        lua_cv::WebSocketTransport::Config ws_cfg;
        ws_cfg.port = config_.ws_port;
        ws_cfg.path = config_.ws_path;
        ws_cfg.max_clients = config_.ws_max_clients;
        ws_ = std::make_unique<lua_cv::WebSocketTransport>(ws_cfg);
        if (!ws_->start()) {
            event("error", MA_EIO, {{"message", "WebSocket server start failed"}});
            ws_.reset();
            rtsp_->stop();
            rtsp_.reset();
            encoder_->shutdown();
            encoder_.reset();
            return MA_EIO;
        }
        // Request IDR when a new browser client connects so it gets a full I-frame.
        auto* enc_ptr = encoder_.get();
        ws_->set_new_client_callback([enc_ptr]() {
            if (enc_ptr) {
                enc_ptr->request_idr();
            }
        });
        event("websocket", MA_OK, {
            {"port", config_.ws_port},
            {"path", config_.ws_path},
            {"codec", "h264"}
        });
    }
#else
    event("error", MA_EINVAL, {{"message", "StreamNode requires CVI MPI support"}});
    return MA_EINVAL;
#endif

    stream_frames_.store(0, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    encode_thread_ = std::thread(&StreamNode::encodeLoop, this);

    return MA_OK;
}

int StreamNode::onStop() {
    if (!running_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

    running_.store(false, std::memory_order_release);

    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }

#ifdef USE_CVI_MPI
    if (rtsp_) {
        rtsp_->stop();
        rtsp_.reset();
    }

    if (ws_) {
        ws_->stop();
        ws_.reset();
    }

    if (encoder_) {
        encoder_->shutdown();
        encoder_.reset();
    }
#endif

    return MA_OK;
}

int StreamNode::onDestroy() {
    onStop();
    camera_node_ = nullptr;
    return MA_OK;
}

int StreamNode::onControl(const std::string& action, const nlohmann::json& /*data*/) {
    if (action == "get_stats") {
        response("get_stats", MA_OK, {
            {"stream_frames", stream_frames_.load(std::memory_order_relaxed)},
            {"ws_frames", ws_frames_.load(std::memory_order_relaxed)},
            {"ws_clients", ws_ ? ws_->client_count() : 0}
        });
        return MA_OK;
    }
    return MA_EINVAL;
}

void StreamNode::encodeLoop() {
#ifdef USE_CVI_MPI
    uint64_t frame_count = 0;
    uint64_t ws_frame_count = 0;
    while (running_.load(std::memory_order_acquire)) {
        if (!encoder_) {
            break;
        }

        lua_cv::VencEncoder::EncodedStream stream;
        if (!encoder_->get_stream(&stream, 100)) {
            continue;
        }

        if (!running_.load(std::memory_order_acquire)) {
            encoder_->release_stream();
            break;
        }

        uint64_t pts = frame_count * (1000000 / static_cast<uint64_t>(config_.fps));
        if (rtsp_ && rtsp_->send_video(stream.data.data(), stream.data.size(), pts)) {
            frame_count++;
            stream_frames_.store(frame_count, std::memory_order_release);
        }

        if (ws_) {
            if (ws_->broadcast_binary(stream.data.data(), stream.data.size())) {
                ws_frame_count++;
                ws_frames_.store(ws_frame_count, std::memory_order_release);
            }
        }

        encoder_->release_stream();
    }
#endif
}

bool StreamNode::resolveStreamChannel(int* vpss_grp, int* vpss_chn) const {
    if (!camera_node_ || !vpss_grp || !vpss_chn) {
        return false;
    }
    return camera_node_->get_stream_binding(vpss_grp, vpss_chn);
}

} // namespace node
