#pragma once

#include "node.h"
#include "error_codes.h"
#include "resource_estimator.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace node {

class NodeServer;
class DataNode;
class CameraNode;

class NodeFactory {
public:
    using CreateFunc = std::function<std::unique_ptr<Node>(
        const std::string& id, const std::string& type)>;

    static NodeFactory& instance();

    // Registration
    void registerNode(const std::string& type,
                      CreateFunc creator,
                      bool singleton = false);

    // Node management
    Node* create(const std::string& id,
                 const std::string& type,
                 const nlohmann::json& config,
                 const std::vector<std::string>& dependencies = {});
    Node* get(const std::string& id);
    int destroy(const std::string& id);
    std::vector<std::string> list() const;

    // Lifecycle control
    int start(const std::string& id);
    int stop(const std::string& id);
    int control(const std::string& id,
                const std::string& action,
                const nlohmann::json& data);

    // Cleanup all nodes
    void destroyAll();

    // Server binding
    void setServer(NodeServer* server) { server_ = server; }
    NodeServer* server() const { return server_; }

    // Error information from last failed operation
    int lastErrorCode() const { return last_error_code_; }
    const std::string& lastErrorReason() const { return last_error_reason_; }

private:
    NodeFactory() = default;

    void collectDependents(Node* node, std::vector<std::string>& out);
    void setupDataFlow(DataNode* node, const std::vector<std::string>& dependencies);
    void teardownDataFlow(DataNode* node);

    struct NodeInfo {
        CreateFunc creator;
        bool singleton = false;
    };

    std::map<std::string, NodeInfo> registry_;
    std::map<std::string, std::unique_ptr<Node>> nodes_;
    std::map<std::string, std::string> singleton_instances_;  // type -> id
    std::map<std::string, std::vector<std::string>> pending_dependencies_;  // node_id -> missing deps
    NodeServer* server_ = nullptr;
    int last_error_code_ = MA_OK;
    std::string last_error_reason_;
    mutable std::mutex mutex_;
};

// Registration macros
#define REGISTER_NODE(type, NodeClass) \
    static struct NodeClass##Registrar { \
        NodeClass##Registrar() { \
            ::node::NodeFactory::instance().registerNode(type, \
                [](const std::string& id, const std::string& t) { \
                    return std::make_unique<NodeClass>(id, t); \
                }); \
        } \
    } g_##NodeClass##Registrar

#define REGISTER_NODE_SINGLETON(type, NodeClass) \
    static struct NodeClass##Registrar { \
        NodeClass##Registrar() { \
            ::node::NodeFactory::instance().registerNode(type, \
                [](const std::string& id, const std::string& t) { \
                    return std::make_unique<NodeClass>(id, t); \
                }, true); \
        } \
    } g_##NodeClass##Registrar

} // namespace node
