#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lua_cv {

class WebSocketTransport {
public:
    enum class PayloadType {
        Binary,
        Text,
    };

    struct Config {
        int port = 8080;
        std::string path = "/";
        int max_clients = 8;
        int poll_ms = 10;
        int queue_depth = 15;  // max frames in broadcast queue (~500ms at 30fps)
    };

    explicit WebSocketTransport(const Config& config);
    ~WebSocketTransport();

    WebSocketTransport(const WebSocketTransport&) = delete;
    WebSocketTransport& operator=(const WebSocketTransport&) = delete;
    WebSocketTransport(WebSocketTransport&&) = delete;
    WebSocketTransport& operator=(WebSocketTransport&&) = delete;

    bool start();
    void stop();
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    bool broadcast_binary(const uint8_t* data, size_t length, bool is_keyframe = false);
    bool broadcast_text(const char* data, size_t length);

    int client_count() const { return client_count_.load(std::memory_order_relaxed); }
    const Config& config() const { return config_; }

    // Called on the mongoose IO thread whenever a new WebSocket client connects.
    // Register encoder_->request_idr() here so new clients get an I-frame immediately.
    void set_new_client_callback(std::function<void()> cb) {
        std::lock_guard<std::mutex> lock(cb_mutex_);
        new_client_cb_ = std::move(cb);
    }

private:
#ifdef USE_MONGOOSE_WS
    struct BroadcastFrame {
        PayloadType type = PayloadType::Binary;
        std::vector<uint8_t> data;
        bool is_keyframe = false;
    };

    struct Impl;
    std::unique_ptr<Impl> impl_;
#endif

    bool enqueue_frame(PayloadType type, const uint8_t* data, size_t length, bool is_keyframe = false);
    void run_loop();

    Config config_;
    std::atomic<bool> running_{false};
    std::atomic<int> client_count_{0};
    std::thread io_thread_;

    mutable std::mutex cb_mutex_;
    std::function<void()> new_client_cb_;

#ifdef USE_MONGOOSE_WS
    mutable std::mutex queue_mutex_;
    std::vector<BroadcastFrame> queue_;
#endif
};

}  // namespace lua_cv
