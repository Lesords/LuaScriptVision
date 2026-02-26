#pragma once

#include "executor.h"
#include "node_factory.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

// Forward declaration for mosquitto
struct mosquitto;
struct mosquitto_message;

namespace node {

// MQTT message types (compatible with node-red-contrib-sscma)
// Values MUST match constants.js: RESPONSE=0, EVENT=1, LOG=2, REQUEST=3, HEARTBEAT=4
enum class MessageType {
    RESPONSE = 0,
    EVENT = 1,
    LOG = 2,
    REQUEST = 3,
    HEARTBEAT = 4
};

class NodeServer {
public:
    struct Config {
        std::string host = "localhost";
        int port = 1883;
        std::string username;
        std::string password;
        std::string client_id = "recamera";
        int keep_alive = 60;
    };

    NodeServer();
    explicit NodeServer(const Config& config);
    ~NodeServer();

    // Non-copyable, non-movable
    NodeServer(const NodeServer&) = delete;
    NodeServer& operator=(const NodeServer&) = delete;
    NodeServer(NodeServer&&) = delete;
    NodeServer& operator=(NodeServer&&) = delete;

    // Lifecycle
    bool start();
    void stop();
    bool isRunning() const { return running_.load(std::memory_order_acquire); }
    bool isConnected() const { return connected_.load(std::memory_order_acquire); }

    // Send messages to client
    void response(const std::string& node_id,
                  const std::string& name,
                  int code,
                  const nlohmann::json& data);
    void event(const std::string& node_id,
               const std::string& name,
               int code,
               const nlohmann::json& data);

    // Config access
    const Config& config() const { return config_; }

private:
    void onConnect(int rc);
    void onDisconnect(int rc);
    void onMessage(const std::string& topic, const std::string& payload);
    void handleRequest(const std::string& node_id, const nlohmann::json& msg);
    int handleCreate(const std::string& node_id, const nlohmann::json& data);
    int handleClear();  // Global command: clear all nodes
    int handleHealth(); // Global command: health check
    std::string extractNodeId(const std::string& topic);

    // Static callbacks for mosquitto
    static void on_connect_cb(struct mosquitto* mosq, void* obj, int rc);
    static void on_disconnect_cb(struct mosquitto* mosq, void* obj, int rc);
    static void on_message_cb(struct mosquitto* mosq, void* obj,
                              const struct mosquitto_message* msg);

    Config config_;
    struct mosquitto* mosq_ = nullptr;
    std::unique_ptr<Executor> executor_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::thread loop_thread_;

    std::string topic_in_prefix_;   // sscma/v0/<client_id>/node/in
    std::string topic_out_prefix_;  // sscma/v0/<client_id>/node/out
};

} // namespace node
