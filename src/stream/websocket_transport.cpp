#include "websocket_transport.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#ifdef USE_MONGOOSE_WS
#include "mongoose.h"
#endif

namespace lua_cv {

#ifdef USE_MONGOOSE_WS
struct WebSocketTransport::Impl {
    explicit Impl(WebSocketTransport* owner_ptr)
        : owner(owner_ptr) {
        std::memset(&mgr, 0, sizeof(mgr));
    }

    static void handle_event(struct mg_connection* c, int ev, void* ev_data) {
        Impl* self = nullptr;
        if (c) {
            self = static_cast<Impl*>(c->fn_data);
            if (!self && c->mgr) {
                self = static_cast<Impl*>(c->mgr->userdata);
            }
        }
        if (!self || !self->owner) {
            return;
        }

        switch (ev) {
            case MG_EV_HTTP_MSG: {
                auto* hm = static_cast<struct mg_http_message*>(ev_data);
                if (!hm) {
                    break;
                }

                const std::string req_uri(hm->uri.buf, hm->uri.len);
                const bool path_ok = self->owner->config_.path.empty() ||
                                     self->owner->config_.path == "/" ||
                                     req_uri == self->owner->config_.path;
                if (!path_ok) {
                    mg_http_reply(c, 404, "", "Not Found\n");
                    c->is_draining = 1;
                    break;
                }

                if (self->owner->client_count() >= self->owner->config_.max_clients) {
                    mg_http_reply(c, 429, "", "Too Many Clients\n");
                    c->is_draining = 1;
                    break;
                }

                mg_ws_upgrade(c, hm, nullptr);
                break;
            }

            case MG_EV_WS_OPEN: {
                self->clients.push_back(c);
                self->owner->client_count_.store(
                    static_cast<int>(self->clients.size()), std::memory_order_relaxed);
                // Request an IDR frame so the new client receives a complete I-frame immediately.
                std::function<void()> cb;
                {
                    std::lock_guard<std::mutex> lock(self->owner->cb_mutex_);
                    cb = self->owner->new_client_cb_;
                }
                if (cb) {
                    cb();
                }
                break;
            }

            case MG_EV_CLOSE: {
                auto it = std::remove(self->clients.begin(), self->clients.end(), c);
                if (it != self->clients.end()) {
                    self->clients.erase(it, self->clients.end());
                    self->owner->client_count_.store(
                        static_cast<int>(self->clients.size()), std::memory_order_relaxed);
                }
                break;
            }

            default:
                break;
        }
    }

    void drain_broadcast_queue() {
        std::vector<BroadcastFrame> local;
        {
            std::lock_guard<std::mutex> lock(owner->queue_mutex_);
            if (owner->queue_.empty()) {
                return;
            }
            local.swap(owner->queue_);
        }

        if (clients.empty()) {
            return;
        }

        for (const auto& frame : local) {
            int opcode = (frame.type == PayloadType::Text)
                             ? WEBSOCKET_OP_TEXT
                             : WEBSOCKET_OP_BINARY;
            for (auto* client : clients) {
                if (!client || client->is_closing) {
                    continue;
                }
                mg_ws_send(client, frame.data.data(), frame.data.size(), opcode);
            }
        }
    }

    WebSocketTransport* owner = nullptr;
    struct mg_mgr mgr;
    struct mg_connection* listener = nullptr;
    std::vector<struct mg_connection*> clients;
};
#endif

WebSocketTransport::WebSocketTransport(const Config& config)
    : config_(config) {
#ifdef USE_MONGOOSE_WS
    impl_ = std::make_unique<Impl>(this);
#endif
}

WebSocketTransport::~WebSocketTransport() {
    stop();
}

bool WebSocketTransport::start() {
    if (running_.load(std::memory_order_acquire)) {
        return true;
    }

#ifndef USE_MONGOOSE_WS
    std::cerr << "[WS] WebSocket backend unavailable (USE_MONGOOSE_WS not set)" << std::endl;
    return false;
#else
    if (!impl_) {
        return false;
    }

    mg_mgr_init(&impl_->mgr);
    impl_->mgr.userdata = impl_.get();
    if (!mg_wakeup_init(&impl_->mgr)) {
        mg_mgr_free(&impl_->mgr);
        std::cerr << "[WS] mg_wakeup_init failed" << std::endl;
        return false;
    }

    std::string listen_url = "http://0.0.0.0:" + std::to_string(config_.port);
    impl_->listener = mg_http_listen(&impl_->mgr, listen_url.c_str(), Impl::handle_event, impl_.get());
    if (!impl_->listener) {
        mg_mgr_free(&impl_->mgr);
        std::cerr << "[WS] listen failed on " << listen_url << std::endl;
        return false;
    }

    running_.store(true, std::memory_order_release);
    io_thread_ = std::thread(&WebSocketTransport::run_loop, this);
    std::cout << "[WS] started on " << listen_url << " path=" << config_.path << std::endl;
    return true;
#endif
}

void WebSocketTransport::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

#ifdef USE_MONGOOSE_WS
    if (impl_) {
        mg_wakeup(&impl_->mgr, impl_->listener ? impl_->listener->id : 1, "x", 1);
    }
#endif

    if (io_thread_.joinable()) {
        io_thread_.join();
    }

#ifdef USE_MONGOOSE_WS
    if (impl_) {
        impl_->clients.clear();
        impl_->listener = nullptr;
        mg_mgr_free(&impl_->mgr);
    }
#endif

    client_count_.store(0, std::memory_order_relaxed);
}

bool WebSocketTransport::broadcast_binary(const uint8_t* data, size_t length) {
    return enqueue_frame(PayloadType::Binary, data, length);
}

bool WebSocketTransport::broadcast_text(const char* data, size_t length) {
    return enqueue_frame(PayloadType::Text,
                         reinterpret_cast<const uint8_t*>(data),
                         length);
}

bool WebSocketTransport::enqueue_frame(PayloadType type,
                                       const uint8_t* data,
                                       size_t length) {
    if (!running_.load(std::memory_order_acquire)) {
        return false;
    }
    if (!data || length == 0) {
        return false;
    }

#ifndef USE_MONGOOSE_WS
    (void)type;
    return false;
#else
    BroadcastFrame frame;
    frame.type = type;
    frame.data.assign(data, data + length);

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Limit queue depth to 3 frames to prevent burst delivery on network jitter.
        // Drop oldest frame when full so the client always gets the most recent data.
        if (queue_.size() >= 3) {
            queue_.erase(queue_.begin());
        }
        queue_.push_back(std::move(frame));
    }

    if (impl_) {
        mg_wakeup(&impl_->mgr, impl_->listener ? impl_->listener->id : 1, "d", 1);
    }
    return true;
#endif
}

void WebSocketTransport::run_loop() {
#ifndef USE_MONGOOSE_WS
    return;
#else
    while (running_.load(std::memory_order_acquire)) {
        mg_mgr_poll(&impl_->mgr, std::max(1, config_.poll_ms));
        impl_->drain_broadcast_queue();
    }
#endif
}

}  // namespace lua_cv
