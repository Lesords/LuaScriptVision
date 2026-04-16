#pragma once

#include "error_codes.h"
#include "resource_limits.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace node {

// Resource requirement for a node
struct ResourceRequirement {
    int vb_infer_blocks = 0;          // Pool 4 blocks needed
    size_t model_memory = 0;          // Model memory in bytes
};

struct ModelTopology {
    std::string camera_id;            // Upstream camera id (empty if unknown)
    std::string upstream_model_id;    // Direct upstream model id (empty if none)
    std::string model_path;           // For memory estimation and session sharing
};

struct CameraPolicy {
    bool frame_skip_enabled = false;
    double infer_fps_limit = 0.0;
};

// Resource evaluation result
struct EstimateResult {
    bool pass = false;
    int error_code = MA_OK;
    std::string reason;
    ResourceRequirement required;
    ResourceRequirement available;
};

// Evaluates device resources before node creation
class ResourceEstimator {
public:
    static ResourceEstimator& instance();

    // Evaluate if a model node can be created
    EstimateResult evaluate_model(
        const nlohmann::json& config,
        const std::vector<std::string>& dependencies);

    // Track resource usage
    void on_node_started(const std::string& node_id,
                         const ResourceRequirement& usage,
                         const std::string& camera_id = std::string(),
                         const std::string& upstream_model_id = std::string(),
                         const std::string& model_path = std::string());
    void on_node_stopped(const std::string& node_id);

    void register_camera(const std::string& node_id,
                         bool frame_skip_enabled,
                         double infer_fps_limit);
    void unregister_camera(const std::string& node_id);

    // Query model memory using CVI Runtime SDK (preferred)
    // Returns actual tensor memory * 3 (for intermediate buffers)
    size_t query_model_memory(const std::string& model_path);

    // Estimate model memory from file size (fallback)
    size_t estimate_model_memory_from_file(const std::string& model_path);

    // Get current usage
    int current_model_count() const;
    int current_vb_usage() const;

private:
    ResourceEstimator() = default;

    // Calculate VB requirement based on topology
    int calculate_vb_requirement(int model_count, bool is_parallel);

    int count_parallel_models(const std::string& camera_id) const;
    int count_serial_chain(const std::string& upstream_model_id) const;
    bool camera_requires_skip(const std::string& camera_id) const;

    // Get available resources
    ResourceRequirement get_available();
    ResourceRequirement get_available_locked() const;

    std::map<std::string, ResourceRequirement> active_nodes_;
    std::map<std::string, ModelTopology> model_topologies_;
    std::map<std::string, CameraPolicy> camera_policies_;
    mutable std::mutex mutex_;
};

} // namespace node
