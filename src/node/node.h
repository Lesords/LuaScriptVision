#pragma once

#include "error_codes.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace node {

class NodeServer;

class Node {
public:
    Node(const std::string& id, const std::string& type);
    virtual ~Node() = default;

    // Lifecycle methods (called by NodeFactory)
    int create(const nlohmann::json& config);
    int start();
    int stop();
    int destroy();
    int control(const std::string& action, const nlohmann::json& data);

    // Template method callbacks (implemented by derived classes)
    virtual int onCreate(const nlohmann::json& config) = 0;
    virtual int onStart() = 0;
    virtual int onStop() = 0;
    virtual int onDestroy() = 0;
    virtual int onControl(const std::string& action, const nlohmann::json& data);

    // Dependency notification callback
    virtual void onDependencyReady(Node* dep);

    // State accessors
    const std::string& id() const { return id_; }
    const std::string& type() const { return type_; }
    bool isCreated() const { return created_.load(std::memory_order_acquire); }
    bool isStarted() const { return started_.load(std::memory_order_acquire); }
    bool isEnabled() const { return enabled_.load(std::memory_order_acquire); }

    // Dependency management (single call establishes bidirectional link)
    void addDependency(Node* node);
    void removeDependency(Node* node);
    bool allDependenciesReady() const;
    const std::map<std::string, Node*>& dependencies() const { return dependencies_; }
    const std::map<std::string, Node*>& dependents() const { return dependents_; }

    // Server communication
    void setServer(NodeServer* server) { server_ = server; }
    NodeServer* server() const { return server_; }
    void response(const std::string& name, int code, const nlohmann::json& data);
    void event(const std::string& name, int code, const nlohmann::json& data);

    // Error information (for node_factory to retrieve)
    const std::string& lastError() const { return last_error_; }

protected:
    std::string id_;
    std::string type_;
    std::atomic<bool> created_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> enabled_{true};
    std::string last_error_;  // Last error message from onCreate/onStart

    std::map<std::string, Node*> dependencies_;
    std::map<std::string, Node*> dependents_;

    NodeServer* server_ = nullptr;
    mutable std::mutex mutex_;
};

} // namespace node
