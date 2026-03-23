#include "node.h"
#include "node_server.h"

#include <iostream>

namespace node {

Node::Node(const std::string& id, const std::string& type)
    : id_(id), type_(type) {}

int Node::create(const nlohmann::json& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (created_.load(std::memory_order_acquire)) {
        return MA_EEXIST;
    }

    int ret = onCreate(config);
    if (ret == MA_OK) {
        created_.store(true, std::memory_order_release);
    }
    return ret;
}

int Node::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!created_.load(std::memory_order_acquire)) {
        return MA_EINVAL;
    }
    if (started_.load(std::memory_order_acquire)) {
        return MA_OK;
    }
    if (!allDependenciesReady()) {
        return MA_EAGAIN;
    }

    int ret = onStart();
    if (ret == MA_OK) {
        started_.store(true, std::memory_order_release);
        // Notify dependents that we are ready
        for (auto& [id, dep] : dependents_) {
            dep->onDependencyReady(this);
        }
    }
    return ret;
}

int Node::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

    // Stop all dependents first (cascade stop)
    for (auto& [id, dep] : dependents_) {
        dep->stop();
    }

    int ret = onStop();
    if (ret == MA_OK) {
        started_.store(false, std::memory_order_release);
    }
    return ret;
}

int Node::destroy() {
    // First stop if running
    stop();

    std::lock_guard<std::mutex> lock(mutex_);
    if (!created_.load(std::memory_order_acquire)) {
        return MA_OK;
    }

    int ret = onDestroy();
    created_.store(false, std::memory_order_release);
    return ret;
}

int Node::control(const std::string& action, const nlohmann::json& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!created_.load(std::memory_order_acquire)) {
        return MA_EINVAL;
    }
    return onControl(action, data);
}

int Node::onControl(const std::string& /*action*/, const nlohmann::json& /*data*/) {
    return MA_EINVAL;  // Not implemented by default
}

void Node::onDependencyReady(Node* /*dep*/) {
    // Try to start if all dependencies are ready
    if (allDependenciesReady() && created_.load(std::memory_order_acquire) &&
        !started_.load(std::memory_order_acquire)) {
        start();
    }
}

void Node::addDependency(Node* node) {
    if (!node || node == this) {
        return;  // Prevent self-reference
    }

    // Lock ordering: always lock the node with smaller address first to avoid deadlock
    // This prevents ABBA deadlock when A->addDependency(B) and B->addDependency(A)
    // are called simultaneously from different threads
    Node* first = this < node ? this : node;
    Node* second = this < node ? node : this;

    std::lock_guard<std::mutex> lock1(first->mutex_);
    std::lock_guard<std::mutex> lock2(second->mutex_);

    dependencies_[node->id()] = node;
    node->dependents_[id_] = this;
}

void Node::removeDependency(Node* node) {
    if (!node) {
        return;
    }

    // Lock ordering: always lock the node with smaller address first
    Node* first = this < node ? this : node;
    Node* second = this < node ? node : this;

    std::lock_guard<std::mutex> lock1(first->mutex_);
    std::lock_guard<std::mutex> lock2(second->mutex_);

    dependencies_.erase(node->id());
    node->dependents_.erase(id_);
}

bool Node::allDependenciesReady() const {
    for (const auto& [id, dep] : dependencies_) {
        if (!dep->isStarted()) {
            return false;
        }
    }
    return true;
}

void Node::response(const std::string& name, int code, const nlohmann::json& data) {
    if (server_) {
        server_->response(id_, name, code, data);
    }
}

void Node::event(const std::string& name, int code, const nlohmann::json& data) {
    // 调试信息：输出事件名称、代码和数据内容
    std::cout << "[Debug] Event: " << name << ", Code: " << code << ", Data: " << data.dump() << std::endl;
    if (server_) {
        server_->event(id_, name, code, data);
    }
}

} // namespace node
