#include "node_factory.h"
#include "node_server.h"
#include "data_node.h"

#ifdef USE_CVI_MPI
#include "camera_node.h"
#endif

namespace node {

NodeFactory& NodeFactory::instance() {
    static NodeFactory inst;
    return inst;
}

void NodeFactory::registerNode(const std::string& type,
                                CreateFunc creator,
                                bool singleton) {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_[type] = {creator, singleton};
}

Node* NodeFactory::create(const std::string& id,
                          const std::string& type,
                          const nlohmann::json& config,
                          const std::vector<std::string>& dependencies) {
    std::unique_lock<std::mutex> lock(mutex_);

    last_error_code_ = MA_OK;
    last_error_reason_.clear();

    // Check if node already exists
    if (nodes_.find(id) != nodes_.end()) {
        last_error_code_ = MA_EEXIST;
        last_error_reason_ = "Node with id '" + id + "' already exists";
        return nullptr;
    }

    // Find creator
    auto it = registry_.find(type);
    if (it == registry_.end()) {
        last_error_code_ = MA_EINVAL;
        last_error_reason_ = "Unknown node type: " + type;
        return nullptr;
    }

    // Check singleton constraint
    if (it->second.singleton) {
        auto sit = singleton_instances_.find(type);
        if (sit != singleton_instances_.end()) {
            last_error_code_ = MA_EEXIST;
            last_error_reason_ = "Singleton node of type '" + type +
                                "' already exists with id '" + sit->second + "'";
            return nullptr;
        }
    }

    // Resource evaluation for model nodes
    if (type == "model") {
        auto estimate = ResourceEstimator::instance().evaluate_model(config, dependencies);
        if (!estimate.pass) {
            last_error_code_ = estimate.error_code;
            last_error_reason_ = estimate.reason;
            return nullptr;
        }
    }

    // Create node
    auto node = it->second.creator(id, type);
    if (!node) {
        last_error_code_ = MA_ENOMEM;
        last_error_reason_ = "Failed to create node instance";
        return nullptr;
    }

    // Set server reference
    node->setServer(server_);

    // Setup dependencies (addDependency establishes bidirectional link)
    // For Node-RED compatibility: don't fail if dependency doesn't exist yet
    // The node will be started when all dependencies become ready
    std::vector<std::string> missing_deps;
    for (const auto& dep_id : dependencies) {
        auto dep = nodes_.find(dep_id);
        if (dep != nodes_.end()) {
            node->addDependency(dep->second.get());
        } else {
            // Dependency doesn't exist yet, record it
            missing_deps.push_back(dep_id);
        }
    }

    // Store pending dependencies if any
    if (!missing_deps.empty()) {
        pending_dependencies_[id] = missing_deps;
    }

    // Call onCreate
    int ret = node->create(config);
    if (ret != MA_OK) {
        last_error_code_ = ret;
        // Try to get detailed error from node's last_error_
        const std::string& node_error = node->lastError();
        if (!node_error.empty()) {
            last_error_reason_ = node_error;
        } else {
            last_error_reason_ = "Node onCreate failed with code " + std::to_string(ret);
        }
        return nullptr;
    }

    // Track singleton
    if (it->second.singleton) {
        singleton_instances_[type] = id;
    }

    Node* ptr = node.get();
    nodes_[id] = std::move(node);

#ifdef USE_CVI_MPI
    if (auto* camera = dynamic_cast<CameraNode*>(ptr)) {
        ResourceEstimator::instance().register_camera(
            id,
            camera->frame_skip_enabled(),
            camera->infer_fps_limit());
    }
#endif

    // Setup data flow for DataNode types (connect inbox to upstream output)
    if (auto* data_node = dynamic_cast<DataNode*>(ptr)) {
        setupDataFlow(data_node, dependencies);
    }

#ifdef USE_CVI_MPI
    // If this node requests stream frame delivery, enable it on all upstream CameraNodes.
    if (config.value("deliverStreamFrame", false)) {
        for (const auto& dep_id : dependencies) {
            auto dep_it = nodes_.find(dep_id);
            if (dep_it == nodes_.end()) continue;
            if (auto* camera = dynamic_cast<CameraNode*>(dep_it->second.get())) {
                camera->set_deliver_stream_frame(true);
            }
        }
    }
#endif

    // Collect nodes that need to be started (to start them outside the factory lock,
    // because onStart() may block for hundreds of ms, e.g. camera open + ISP warmup)
    std::vector<Node*> to_start;

    if (ptr->allDependenciesReady()) {
        to_start.push_back(ptr);
    }

    // Check if this newly created node is a pending dependency for any other nodes
    // This handles the case where dependent nodes are created before their dependencies
    for (auto& [other_id, other_node] : nodes_) {
        if (other_id == id) continue;  // Skip self

        auto pending_it = pending_dependencies_.find(other_id);
        if (pending_it != pending_dependencies_.end()) {
            auto& pending_deps = pending_it->second;
            auto dep_it = std::find(pending_deps.begin(), pending_deps.end(), id);
            if (dep_it != pending_deps.end()) {
                // This node was a pending dependency, establish the link now
                other_node->addDependency(ptr);
                pending_deps.erase(dep_it);

                // Setup data flow if the other node is a DataNode
                if (auto* data_node = dynamic_cast<DataNode*>(other_node.get())) {
                    std::vector<std::string> deps = {id};
                    setupDataFlow(data_node, deps);
                }

                // If all pending dependencies are resolved, queue start
                if (pending_deps.empty()) {
                    pending_dependencies_.erase(pending_it);
                    if (other_node->allDependenciesReady() && !other_node->isStarted()) {
                        to_start.push_back(other_node.get());
                    }
                }
            }
        }
    }

    // Release factory lock before calling start() so that long-running onStart()
    // (e.g. CameraNode: sensor init + 300ms ISP warmup) doesn't block MQTT message handling
    lock.unlock();

    for (Node* n : to_start) {
        n->start();
    }

    return ptr;
}

Node* NodeFactory::get(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

int NodeFactory::destroy(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return MA_ENOENT;
    }

    Node* node = it->second.get();

    // Collect all dependents first (recursive)
    std::vector<std::string> to_destroy;
    collectDependents(node, to_destroy);

    // Destroy dependents first (reverse dependency order)
    for (const auto& dep_id : to_destroy) {
        auto dep_it = nodes_.find(dep_id);
        if (dep_it != nodes_.end()) {
            // Teardown data flow before destroying
            if (auto* data_node = dynamic_cast<DataNode*>(dep_it->second.get())) {
                teardownDataFlow(data_node);
            }

            dep_it->second->destroy();

            // Update singleton tracking
            const auto& dep_type = dep_it->second->type();
            auto reg_it = registry_.find(dep_type);
            if (reg_it != registry_.end() && reg_it->second.singleton) {
                singleton_instances_.erase(dep_type);
            }

            // Resource tracking
            ResourceEstimator::instance().on_node_stopped(dep_id);

            nodes_.erase(dep_it);
        }
    }

    // Teardown data flow for this node
    if (auto* data_node = dynamic_cast<DataNode*>(node)) {
        teardownDataFlow(data_node);
    }

    // Destroy this node
    node->destroy();

    // Update singleton tracking
    auto reg_it = registry_.find(node->type());
    if (reg_it != registry_.end() && reg_it->second.singleton) {
        singleton_instances_.erase(node->type());
    }

    // Resource tracking
    ResourceEstimator::instance().on_node_stopped(id);
    if (node->type() == "camera") {
        ResourceEstimator::instance().unregister_camera(id);
    }

    nodes_.erase(it);

    // Clean up pending dependencies for this node
    pending_dependencies_.erase(id);

    // Also remove this node from other nodes' pending dependencies
    for (auto& [node_id, deps] : pending_dependencies_) {
        deps.erase(std::remove(deps.begin(), deps.end(), id), deps.end());
    }

    return MA_OK;
}

std::vector<std::string> NodeFactory::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(nodes_.size());
    for (const auto& [id, node] : nodes_) {
        ids.push_back(id);
    }
    return ids;
}

int NodeFactory::start(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return MA_ENOENT;
    }
    return it->second->start();
}

int NodeFactory::stop(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return MA_ENOENT;
    }
#ifdef USE_CVI_MPI
    if (auto* camera = dynamic_cast<CameraNode*>(it->second.get())) {
        camera->stopCapture();
    }
#endif
    return it->second->stop();
}

int NodeFactory::control(const std::string& id,
                         const std::string& action,
                         const nlohmann::json& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return MA_ENOENT;
    }
    return it->second->control(action, data);
}

void NodeFactory::destroyAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Stop and destroy all nodes
#ifdef USE_CVI_MPI
    for (auto& [id, node] : nodes_) {
        if (auto* camera = dynamic_cast<CameraNode*>(node.get())) {
            camera->stopCapture();
        }
    }
#endif
    for (auto& [id, node] : nodes_) {
        node->destroy();
        ResourceEstimator::instance().on_node_stopped(id);
        if (node->type() == "camera") {
            ResourceEstimator::instance().unregister_camera(id);
        }
    }

    nodes_.clear();
    singleton_instances_.clear();
}

#ifdef USE_CVI_MPI
CameraNode* NodeFactory::find_camera_node() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, node] : nodes_) {
        if (node->type() == "camera" && node->isStarted()) {
            return static_cast<CameraNode*>(node.get());
        }
    }
    return nullptr;
}
#else
CameraNode* NodeFactory::find_camera_node() { return nullptr; }
#endif

void NodeFactory::collectDependents(Node* node, std::vector<std::string>& out) {
    for (const auto& [dep_id, dep] : node->dependents()) {
        // Recursively collect dependents
        collectDependents(dep, out);
        out.push_back(dep_id);
    }
}

void NodeFactory::setupDataFlow(DataNode* node, const std::vector<std::string>& dependencies) {
    // Connect this node's inbox to upstream node's output
    // DataNode subscribes to its dependencies' outputs
    for (const auto& dep_id : dependencies) {
        auto dep_it = nodes_.find(dep_id);
        if (dep_it == nodes_.end()) continue;

        Node* dep = dep_it->second.get();

#ifdef USE_CVI_MPI
        // Case 1: Upstream is CameraNode - use channel-based attach
        if (auto* camera = dynamic_cast<CameraNode*>(dep)) {
            // ModelNode subscribes to inference channel
            camera->attach(FrameChannel::INFER, node->inbox());
        }
        // Case 2: Upstream is another DataNode (e.g., ModelNode) - use generic attach
        else
#endif
        if (auto* upstream_data = dynamic_cast<DataNode*>(dep)) {
            upstream_data->attach(node->inbox());
        }
    }
}

void NodeFactory::teardownDataFlow(DataNode* node) {
    // Disconnect this node's inbox from upstream nodes
    for (const auto& [dep_id, dep] : node->dependencies()) {
#ifdef USE_CVI_MPI
        // Case 1: Upstream is CameraNode
        if (auto* camera = dynamic_cast<CameraNode*>(dep)) {
            camera->detach(FrameChannel::INFER, node->inbox());
        }
        // Case 2: Upstream is another DataNode
        else
#endif
        if (auto* upstream_data = dynamic_cast<DataNode*>(dep)) {
            upstream_data->detach(node->inbox());
        }
    }
}

} // namespace node
