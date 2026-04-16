#include "resource_estimator.h"

#include <sys/stat.h>
#include <set>

#ifdef USE_CVI_TPU
#include <cviruntime.h>
#endif

// Weak symbols for SessionManager integration.
// When session_manager.cpp is linked (node_server, parallel_infer_stream), strong versions take over.
// When not linked (test targets), these safe defaults are used.
#ifdef USE_CVI_TPU
bool __attribute__((weak)) session_manager_is_loaded(const std::string& path) {
    (void)path;
    return false;
}
size_t __attribute__((weak)) session_manager_total_memory() {
    return 0;
}
#endif

namespace node {

ResourceEstimator& ResourceEstimator::instance() {
    static ResourceEstimator inst;
    return inst;
}

EstimateResult ResourceEstimator::evaluate_model(
    const nlohmann::json& config,
    const std::vector<std::string>& dependencies) {

    std::lock_guard<std::mutex> lock(mutex_);

    EstimateResult result;
    result.available = get_available_locked();

    // Determine topology (parallel vs serial) based on dependencies
    std::string upstream_model_id;
    std::string camera_id;

    for (const auto& dep_id : dependencies) {
        if (model_topologies_.count(dep_id) > 0) {
            upstream_model_id = dep_id;
        }
        if (camera_policies_.count(dep_id) > 0) {
            camera_id = dep_id;
        }
    }

    if (camera_id.empty() && !upstream_model_id.empty()) {
        auto it = model_topologies_.find(upstream_model_id);
        if (it != model_topologies_.end()) {
            camera_id = it->second.camera_id;
        }
    }

    bool is_parallel = upstream_model_id.empty() && !camera_id.empty();
    bool is_serial = !upstream_model_id.empty();

    int new_model_count = static_cast<int>(active_nodes_.size()) + 1;
    int parallel_count = is_parallel ? count_parallel_models(camera_id) : 0;
    int new_parallel_count = is_parallel ? (parallel_count + 1) : 0;
    int serial_chain_len = is_serial ? count_serial_chain(upstream_model_id) : 0;
    int new_serial_count = is_serial ? (serial_chain_len + 1) : new_model_count;

    // Query model memory if path provided
    bool model_already_loaded = false;
    if (config.contains("model")) {
        std::string model_path = config["model"];
#ifdef USE_CVI_TPU
        model_already_loaded = session_manager_is_loaded(model_path);
#endif
        if (model_already_loaded) {
            // Session already loaded by another node — no additional memory cost
            result.required.model_memory = 0;
        } else {
            result.required.model_memory = query_model_memory(model_path);
            if (result.required.model_memory == 0) {
                result.required.model_memory = estimate_model_memory_from_file(model_path);
            }
        }
    }

    // Check ION memory constraint (actual TPU memory, not just model count)
    if (!model_already_loaded && result.required.model_memory > 0) {
#ifdef USE_CVI_TPU
        size_t current_tpu_mem = session_manager_total_memory();
        size_t total_after = current_tpu_mem + result.required.model_memory;
        size_t available_bytes = ION_AVAILABLE_MB * 1024 * 1024;
        if (total_after > available_bytes) {
            result.pass = false;
            result.error_code = MA_ENOMEM;
            result.reason = "ION memory insufficient: need " +
                           std::to_string(total_after / 1024 / 1024) + "MB total" +
                           " (current: " + std::to_string(current_tpu_mem / 1024 / 1024) +
                           "MB + new: " + std::to_string(result.required.model_memory / 1024 / 1024) +
                           "MB), available " + std::to_string(ION_AVAILABLE_MB) + "MB";
            return result;
        }
#endif
    }

    // VB pool constraint — each concurrent inference needs its own input frame buffer
    // This is about frame buffers, not model weights, so it applies regardless of session sharing
    // However, if the model session is already loaded (shared via SessionManager), the
    // memory impact is negligible, so we skip the hard parallel count limit.
    if (!model_already_loaded) {
        if (is_parallel && new_parallel_count >= MAX_PARALLEL_MODELS &&
            !camera_requires_skip(camera_id)) {
            result.pass = false;
            result.error_code = MA_EINVAL;
            result.reason = "Parallel mode with " +
                           std::to_string(new_parallel_count) +
                           " models requires camera infer_fps_limit";
            return result;
        }
    }

    int vb_model_count = is_parallel ? new_parallel_count : new_serial_count;
    result.required.vb_infer_blocks = calculate_vb_requirement(vb_model_count, is_parallel);

    if (is_parallel && new_parallel_count >= MAX_PARALLEL_MODELS &&
        camera_requires_skip(camera_id)) {
        result.required.vb_infer_blocks = 1 + new_parallel_count;
    }

    if (result.required.vb_infer_blocks > VB_POOL4_TOTAL) {
        // Allow overflow for shared sessions: if the model is already loaded,
        // the extra VB blocks are for frame buffers only, which can be managed
        // by the TPU scheduler's serialization.
        if (!model_already_loaded) {
            result.pass = false;
            result.error_code = MA_ENOMEM;
            result.reason = "VB Pool insufficient: need " +
                           std::to_string(result.required.vb_infer_blocks) +
                           " blocks, available " + std::to_string(VB_POOL4_TOTAL);
            return result;
        }
    }

    result.pass = true;
    result.error_code = MA_OK;
    return result;
}

void ResourceEstimator::on_node_started(const std::string& node_id,
                                         const ResourceRequirement& usage,
                                         const std::string& camera_id,
                                         const std::string& upstream_model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_nodes_[node_id] = usage;
    model_topologies_[node_id] = {camera_id, upstream_model_id};
}

void ResourceEstimator::on_node_stopped(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_nodes_.erase(node_id);
    model_topologies_.erase(node_id);
}

void ResourceEstimator::register_camera(const std::string& node_id,
                                        bool frame_skip_enabled,
                                        double infer_fps_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    camera_policies_[node_id] = {frame_skip_enabled, infer_fps_limit};
}

void ResourceEstimator::unregister_camera(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    camera_policies_.erase(node_id);
}

size_t ResourceEstimator::query_model_memory(const std::string& model_path) {
#ifdef USE_CVI_TPU
    struct stat st;
    if (stat(model_path.c_str(), &st) != 0) {
        return 0;
    }

    // Use CVI Runtime SDK to query actual model memory usage
    CVI_MODEL_HANDLE model = nullptr;
    CVI_RC ret = CVI_NN_RegisterModel(model_path.c_str(), &model);
    if (ret != CVI_RC_SUCCESS) {
        return 0;  // Fall back to file size estimation
    }

    CVI_TENSOR* inputs = nullptr;
    CVI_TENSOR* outputs = nullptr;
    int32_t input_num = 0, output_num = 0;

    ret = CVI_NN_GetInputOutputTensors(model, &inputs, &input_num, &outputs, &output_num);
    if (ret != CVI_RC_SUCCESS) {
        CVI_NN_CleanupModel(model);
        return 0;
    }

    // Calculate total tensor memory
    size_t total_size = 0;

    // Sum input tensor sizes
    for (int i = 0; i < input_num; i++) {
        total_size += CVI_NN_TensorSize(&inputs[i]);
    }

    // Sum output tensor sizes
    for (int i = 0; i < output_num; i++) {
        total_size += CVI_NN_TensorSize(&outputs[i]);
    }

    // Note: This is only tensor size, not including model weights and intermediate buffers
    // Actual memory usage is typically 2-3x of tensor size
    // Per design doc: use 3x multiplier
    size_t estimated_total = total_size * 3;

    CVI_NN_CleanupModel(model);
    return estimated_total;
#else
    // No TPU support, return 0 to trigger fallback
    (void)model_path;
    return 0;
#endif
}

size_t ResourceEstimator::estimate_model_memory_from_file(const std::string& model_path) {
    // Estimate based on file size
    // .cvimodel file size ~ weight size
    // Runtime memory ~ file size * 1.5 (including intermediate buffers)
    struct stat st;
    if (stat(model_path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<size_t>(st.st_size * 1.5);
}

int ResourceEstimator::current_model_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(active_nodes_.size());
}

int ResourceEstimator::current_vb_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int total = 0;
    for (const auto& [id, usage] : active_nodes_) {
        total += usage.vb_infer_blocks;
    }
    return total;
}

int ResourceEstimator::calculate_vb_requirement(int model_count, bool is_parallel) {
    if (model_count == 0) {
        return 0;
    }

    if (is_parallel) {
        // Parallel: Camera(1) + N models holding frames + buffer(1)
        return 1 + model_count + VB_BUFFER_RESERVE;
    } else {
        // Serial: Camera(1) + current frame(1) + buffer(1)
        // Frames are passed sequentially, so only need 3 blocks total
        return 3;
    }
}

ResourceRequirement ResourceEstimator::get_available() {
    std::lock_guard<std::mutex> lock(mutex_);
    return get_available_locked();
}

ResourceRequirement ResourceEstimator::get_available_locked() const {
    ResourceRequirement avail;
    int total = 0;
    for (const auto& [id, usage] : active_nodes_) {
        total += usage.vb_infer_blocks;
    }
    avail.vb_infer_blocks = VB_POOL4_TOTAL - total;
    return avail;
}

int ResourceEstimator::count_parallel_models(const std::string& camera_id) const {
    int count = 0;
    for (const auto& [id, topo] : model_topologies_) {
        if (topo.upstream_model_id.empty() && topo.camera_id == camera_id) {
            count++;
        }
    }
    return count;
}

int ResourceEstimator::count_serial_chain(const std::string& upstream_model_id) const {
    if (upstream_model_id.empty()) {
        return 0;
    }

    int count = 0;
    std::string current = upstream_model_id;
    std::set<std::string> visited;

    while (!current.empty()) {
        if (visited.count(current) > 0) {
            break;
        }
        visited.insert(current);
        count++;

        auto it = model_topologies_.find(current);
        if (it == model_topologies_.end()) {
            break;
        }
        current = it->second.upstream_model_id;
    }

    return count;
}

bool ResourceEstimator::camera_requires_skip(const std::string& camera_id) const {
    if (camera_id.empty()) {
        return false;
    }
    auto it = camera_policies_.find(camera_id);
    if (it == camera_policies_.end()) {
        return false;
    }
    return it->second.frame_skip_enabled && it->second.infer_fps_limit > 0.0;
}

} // namespace node
