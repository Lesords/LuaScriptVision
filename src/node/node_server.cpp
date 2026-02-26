#include "node_server.h"

#include <iostream>
#include <mosquitto.h>
#include <sys/stat.h>
#include <vector>

namespace node {

NodeServer::NodeServer()
    : NodeServer(Config{}) {
}

NodeServer::NodeServer(const Config& config)
    : config_(config) {
    // Initialize topic prefixes
    topic_in_prefix_ = "sscma/v0/" + config_.client_id + "/node/in";
    topic_out_prefix_ = "sscma/v0/" + config_.client_id + "/node/out";

    // Initialize mosquitto library (safe to call multiple times)
    mosquitto_lib_init();
}

NodeServer::~NodeServer() {
    stop();
    mosquitto_lib_cleanup();
}

bool NodeServer::start() {
    if (running_.load(std::memory_order_acquire)) {
        return true;
    }

    // Create mosquitto client
    mosq_ = mosquitto_new(config_.client_id.c_str(), true, this);
    if (!mosq_) {
        return false;
    }

    // Set callbacks
    mosquitto_connect_callback_set(mosq_, on_connect_cb);
    mosquitto_disconnect_callback_set(mosq_, on_disconnect_cb);
    mosquitto_message_callback_set(mosq_, on_message_cb);

    // Set auto-reconnect parameters
    mosquitto_reconnect_delay_set(mosq_, 2, 30, true);

    // Set credentials if provided
    if (!config_.username.empty()) {
        mosquitto_username_pw_set(mosq_,
            config_.username.c_str(),
            config_.password.c_str());
    }

    // Connect to broker
    int rc = mosquitto_connect(mosq_,
        config_.host.c_str(),
        config_.port,
        config_.keep_alive);
    if (rc != MOSQ_ERR_SUCCESS) {
        mosquitto_destroy(mosq_);
        mosq_ = nullptr;
        return false;
    }

    // Create executor for async command processing
    executor_ = std::make_unique<Executor>("node_server");

    // Bind factory to server
    NodeFactory::instance().setServer(this);

    running_.store(true, std::memory_order_release);

    // Start mosquitto loop in separate thread
    loop_thread_ = std::thread([this]() {
        mosquitto_loop_forever(mosq_, -1, 1);
    });

    return true;
}

void NodeServer::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);

    // Stop mosquitto loop
    if (mosq_) {
        mosquitto_disconnect(mosq_);
    }

    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }

    // Cleanup
    if (executor_) {
        executor_->cancel();
        executor_.reset();
    }

    // Destroy all nodes
    NodeFactory::instance().destroyAll();
    NodeFactory::instance().setServer(nullptr);

    if (mosq_) {
        mosquitto_destroy(mosq_);
        mosq_ = nullptr;
    }

    connected_.store(false, std::memory_order_release);
}

void NodeServer::response(const std::string& node_id,
                          const std::string& name,
                          int code,
                          const nlohmann::json& data) {
    if (!connected_.load(std::memory_order_acquire)) {
        return;  // Silently drop if not connected
    }

    nlohmann::json msg;
    msg["type"] = static_cast<int>(MessageType::RESPONSE);
    msg["name"] = name;
    msg["code"] = code;

    // Node-RED compatibility: error responses should have string data
    // For non-error codes (MA_OK), data can be object or string
    if (code != MA_OK) {
        if (data.is_object() && data.contains("message")) {
            msg["data"] = data["message"].get<std::string>();
        } else if (data.is_string()) {
            msg["data"] = data.get<std::string>();
        } else {
            msg["data"] = data.dump();  // Fallback: serialize to JSON string
        }
    } else {
        msg["data"] = data;
    }

    // Debug logging for all responses
    std::cout << "[NodeServer] Response: node_id=" << node_id
              << ", name=" << name
              << ", code=" << code
              << ", data=" << msg["data"].dump()
              << ", payload=" << msg.dump() << std::endl;

    // For global commands (empty node_id), use "server" as the node_id
    // This ensures the topic format is valid for MQTT subscriptions
    std::string topic_node_id = node_id.empty() ? "server" : node_id;
    std::string topic = topic_out_prefix_ + "/" + topic_node_id;
    std::string payload = msg.dump();

    mosquitto_publish(mosq_, nullptr, topic.c_str(),
                      static_cast<int>(payload.size()),
                      payload.data(), 0, false);
}

void NodeServer::event(const std::string& node_id,
                       const std::string& name,
                       int code,
                       const nlohmann::json& data) {
    if (!connected_.load(std::memory_order_acquire)) {
        std::cout << "[NodeServer] Event dropped (not connected): node_id=" << node_id
                  << ", name=" << name << std::endl;
        return;  // Silently drop if not connected
    }

    nlohmann::json msg;
    msg["type"] = static_cast<int>(MessageType::EVENT);
    msg["name"] = name;
    msg["code"] = code;

    // Node-RED compatibility: error events should have string data, not object
    // This prevents "[object Object]" display issues in Node-RED frontend
    if (name == "error") {
        if (data.is_object() && data.contains("message")) {
            msg["data"] = data["message"].get<std::string>();
        } else if (data.is_string()) {
            msg["data"] = data.get<std::string>();
        } else {
            msg["data"] = data.dump();  // Fallback: serialize to JSON string
        }
    } else {
        msg["data"] = data;
    }

    // Debug logging for all events
    std::cout << "[NodeServer] Event: node_id=" << node_id
              << ", name=" << name
              << ", code=" << code
              << ", data=" << msg["data"].dump()
              << ", payload=" << msg.dump() << std::endl;

    // For global events (empty node_id), use "server" as the node_id
    std::string topic_node_id = node_id.empty() ? "server" : node_id;
    std::string topic = topic_out_prefix_ + "/" + topic_node_id;
    std::string payload = msg.dump();

    mosquitto_publish(mosq_, nullptr, topic.c_str(),
                      static_cast<int>(payload.size()),
                      payload.data(), 0, false);
}

void NodeServer::onConnect(int rc) {
    if (rc == MOSQ_ERR_SUCCESS) {
        connected_.store(true, std::memory_order_release);

        // Subscribe to input topics
        std::string topic = topic_in_prefix_ + "/+";
        mosquitto_subscribe(mosq_, nullptr, topic.c_str(), 0);

        // Send connection acknowledgement (compatible with sscma-node)
        response("", "node", MA_OK, "");
    }
}

void NodeServer::onDisconnect(int /*rc*/) {
    connected_.store(false, std::memory_order_release);
    // mosquitto_loop will auto-reconnect
}

void NodeServer::onMessage(const std::string& topic, const std::string& payload) {
    // Extract node_id from topic
    std::string node_id = extractNodeId(topic);
    if (node_id.empty()) {
        return;
    }

    // Parse JSON
    nlohmann::json msg;
    try {
        msg = nlohmann::json::parse(payload);
    } catch (...) {
        return;  // Invalid JSON
    }

    // Submit to executor for async processing
    executor_->submit([this, node_id, msg]() {
        handleRequest(node_id, msg);
        return false;  // Don't requeue
    });
}

void NodeServer::handleRequest(const std::string& node_id, const nlohmann::json& msg) {
    int type = msg.value("type", 0);
    if (type != static_cast<int>(MessageType::REQUEST)) {
        return;  // Ignore non-request messages
    }

    std::string action = msg.value("name", "");
    nlohmann::json data = msg.value("data", nlohmann::json::object());

    // Debug logging
    std::cout << "[NodeServer] Request: node_id=" << node_id
              << ", action=" << action
              << ", data=" << data.dump() << std::endl;

    int code = MA_OK;
    nlohmann::json result;

    try {
        if (action == "create") {
            // handleCreate sends its own response for both model and camera nodes
            // and returns MA_OK on success
            code = handleCreate(node_id, data);
            if (code != MA_OK) {
                // On error, data should be a string (not an object)
                // for Node-RED compatibility
                std::string error = NodeFactory::instance().lastErrorReason();
                std::cout << "[NodeServer] Create failed: " << error << std::endl;
                response(node_id, action, code, error);
                return;
            }
            // handleCreate already sent the response for model and camera nodes
            return;  // Don't send another response
        } else if (action == "destroy") {
            code = NodeFactory::instance().destroy(node_id);
        } else if (action == "clear") {
            // Global command: clear all nodes (node_id may be empty)
            code = handleClear();
            std::cout << "[NodeServer] Cleared all nodes" << std::endl;
        } else if (action == "health") {
            // Global command: health check
            code = handleHealth();
        } else if (action == "start") {
            code = NodeFactory::instance().start(node_id);
        } else if (action == "stop") {
            code = NodeFactory::instance().stop(node_id);
        } else if (action == "list") {
            auto ids = NodeFactory::instance().list();
            result["nodes"] = ids;
        } else if (action == "enabled") {
            // Node-RED camera/model node enable/disable command
            // This is handled by NodeFactory::control which calls node->onControl
            // The node will send "enabled" event after state change
            code = NodeFactory::instance().control(node_id, action, data);
        } else if (action == "pause" || action == "light") {
            // Optional Node-RED camera commands
            // These may return MA_ENOTSUP if not implemented
            code = NodeFactory::instance().control(node_id, action, data);
        } else {
            // Other control commands (config, set_fps, etc.)
            code = NodeFactory::instance().control(node_id, action, data);
        }
    } catch (const std::exception& e) {
        code = MA_EINVAL;
        // On exception, data should be a string for Node-RED
        std::string error = e.what();
        std::cout << "[NodeServer] Exception: " << error << std::endl;
        response(node_id, action, code, error);
        return;
    }

    response(node_id, action, code, result);
}

int NodeServer::handleCreate(const std::string& node_id, const nlohmann::json& data) {
    std::string type = data.value("type", "");
    if (type.empty()) {
        std::cout << "[NodeServer] Create failed: missing type field" << std::endl;
        return MA_EINVAL;
    }

    nlohmann::json config = data.value("config", nlohmann::json::object());

    // Debug logging
    std::cout << "[NodeServer] Creating node: id=" << node_id
              << ", type=" << type
              << ", config=" << config.dump()
              << ", dependencies=" << (data.contains("dependencies") ? data["dependencies"].dump() : "[]")
              << std::endl;

    // Node-RED compatibility: auto-add script field for model nodes if not provided
    if (type == "model" && !config.contains("script")) {
        std::string model_path;
        if (config.contains("uri")) {
            model_path = config["uri"].get<std::string>();
        } else if (config.contains("model")) {
            model_path = config["model"].get<std::string>();
        } else {
            // No model path either, will fail later in ModelNode::onCreate
        }

        // Try to auto-detect script path from model name
        if (!model_path.empty()) {
            std::string model_name;
            size_t last_slash = model_path.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string filename = model_path.substr(last_slash + 1);
                size_t dot_pos = filename.find_last_of(".");
                if (dot_pos != std::string::npos) {
                    model_name = filename.substr(0, dot_pos);
                } else {
                    model_name = filename;
                }
            }

            // Default script search paths
            std::vector<std::string> search_paths = {
                "/userdata/Scripts/" + model_name + "_detector.lua",
                "/userdata/Scripts/" + model_name + ".lua",
                "/usr/local/share/luascriptvision/scripts/" + model_name + "_detector.lua",
                "./scripts/" + model_name + "_detector.lua",
                "./scripts/yolo11_tensor_detector.lua",  // Prefer tensor version (CVI-compatible)
                "./scripts/yolo11_detector.lua"          // Legacy fallback
            };

            // Find first existing script
            for (const auto& path : search_paths) {
                struct stat st {};
                if (stat(path.c_str(), &st) == 0) {
                    config["script"] = path;
                    break;
                }
            }

            // If no script found, use a default and let ModelNode handle the error
            if (!config.contains("script")) {
                config["script"] = "/userdata/Scripts/yolo11_detector.lua";
            }
        }
    }

    std::vector<std::string> deps;

    if (data.contains("dependencies")) {
        for (const auto& d : data["dependencies"]) {
            deps.push_back(d.get<std::string>());
        }
    }

    Node* node = NodeFactory::instance().create(node_id, type, config, deps);
    if (!node) {
        return NodeFactory::instance().lastErrorCode();
    }

    // Send node-specific create response with actual data
    // This is compatible with sscma-node which returns node-specific info
    if (type == "model") {
        // Send model info response
        nlohmann::json model_info = {
            {"model_id", "0"},
            {"classes", nlohmann::json::array()}
        };

        if (config.contains("uri")) {
            model_info["model_path"] = config["uri"].get<std::string>();
        } else if (config.contains("model")) {
            model_info["model_path"] = config["model"].get<std::string>();
        }

        response(node_id, "create", MA_OK, model_info);
    } else if (type == "camera") {
        // Send camera info response (compatible with sscma-node)
        nlohmann::json camera_info = {
            {"width", 1920},  // Default, should get from actual config
            {"height", 1080},
            {"fps", 30}
        };

        // Try to get actual values from config
        if (config.contains("width")) {
            camera_info["width"] = config["width"].get<int>();
        }
        if (config.contains("height")) {
            camera_info["height"] = config["height"].get<int>();
        }
        if (config.contains("fps")) {
            camera_info["fps"] = config["fps"].get<double>();
        }
        // Handle option field (0=1080p, 1=720p, 2=480p)
        if (config.contains("option")) {
            int option = config["option"].get<int>();
            switch (option) {
                case 0: camera_info["width"] = 1920; camera_info["height"] = 1080; break;
                case 1: camera_info["width"] = 1280; camera_info["height"] = 720; break;
                case 2: camera_info["width"] = 640; camera_info["height"] = 480; break;
            }
        }

        response(node_id, "create", MA_OK, camera_info);
    }

    return MA_OK;
}

int NodeServer::handleClear() {
    NodeFactory::instance().destroyAll();
    return MA_OK;
}

int NodeServer::handleHealth() {
    // Health check - just return OK
    return MA_OK;
}

std::string NodeServer::extractNodeId(const std::string& topic) {
    // Topic format: sscma/v0/<client>/node/in/<node_id>
    // Find last '/' and extract node_id
    size_t pos = topic.rfind('/');
    if (pos == std::string::npos || pos == topic.length() - 1) {
        return "";
    }
    return topic.substr(pos + 1);
}

// Static callbacks
void NodeServer::on_connect_cb(struct mosquitto* /*mosq*/, void* obj, int rc) {
    auto* server = static_cast<NodeServer*>(obj);
    server->onConnect(rc);
}

void NodeServer::on_disconnect_cb(struct mosquitto* /*mosq*/, void* obj, int rc) {
    auto* server = static_cast<NodeServer*>(obj);
    server->onDisconnect(rc);
}

void NodeServer::on_message_cb(struct mosquitto* /*mosq*/, void* obj,
                                const struct mosquitto_message* msg) {
    auto* server = static_cast<NodeServer*>(obj);
    std::string topic(msg->topic);
    std::string payload(static_cast<char*>(msg->payload), msg->payloadlen);
    server->onMessage(topic, payload);
}

} // namespace node
