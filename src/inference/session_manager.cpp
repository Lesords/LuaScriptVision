#include "session_manager.h"

#ifdef USE_CVI_TPU

#include "cvi_session.h"

#include <cviruntime.h>
#include <iostream>
#include <stdexcept>

// Strong overrides for weak symbols declared in resource_estimator.cpp.
// When session_manager.cpp is linked, these take precedence over the weak defaults.

bool session_manager_is_loaded(const std::string& path) {
    return inference::SessionManager::instance().is_loaded(path);
}

size_t session_manager_total_memory() {
    return inference::SessionManager::instance().total_tpu_memory();
}

namespace inference {

SessionManager& SessionManager::instance() {
    static SessionManager inst;
    return inst;
}

SessionManager::SessionManager() = default;

SessionManager::~SessionManager() {
    // Release all sessions first
    sessions_.clear();
    ref_counts_.clear();

    // Global RT handle cleanup (process exit)
    if (rt_initialized_ && rt_handle_) {
        CVI_RT_DeInit(static_cast<CVI_RT_HANDLE>(rt_handle_));
        rt_handle_ = nullptr;
        rt_initialized_ = false;
    }
}

std::shared_ptr<CviSession> SessionManager::acquire(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Initialize global RT handle on first use
    if (!rt_initialized_) {
        CVI_RT_HANDLE handle = nullptr;
        CVI_RC rc = CVI_RT_Init(&handle);
        if (rc != CVI_RC_SUCCESS) {
            throw std::runtime_error("SessionManager: CVI_RT_Init failed: " + std::to_string(rc));
        }
        rt_handle_ = handle;
        rt_initialized_ = true;
    }

    // Return existing session if already loaded
    auto it = sessions_.find(model_path);
    if (it != sessions_.end()) {
        ref_counts_[model_path]++;
        // Debug: No need to create a new session for the current model, just use the reference directly
        std::cout << "[SessionManager] Reusing session for model: " << model_path
                  << " total count: " << ref_counts_[model_path] << std::endl;
        return it->second;
    }

    // Load new session with shared RT handle
    auto rt = static_cast<CVI_RT_HANDLE>(rt_handle_);
    auto session = std::make_shared<CviSession>(model_path, rt);
    sessions_[model_path] = session;
    ref_counts_[model_path] = 1;

    // Track actual tensor memory (input + output buffers, 3x for intermediate)
    size_t model_mem = 0;
    for (int32_t i = 0; i < session->input_count(); ++i) {
        auto shape = session->get_input_shape(static_cast<size_t>(i));
        size_t elems = 1;
        for (auto d : shape) elems *= static_cast<size_t>(d);
        model_mem += elems * sizeof(float);
    }
    for (int32_t i = 0; i < session->output_count(); ++i) {
        auto shape = session->get_output_shape(static_cast<size_t>(i));
        size_t elems = 1;
        for (auto d : shape) elems *= static_cast<size_t>(d);
        model_mem += elems * sizeof(float);
    }
    total_memory_ += model_mem * 3;

    std::cout << "[SessionManager] Loaded model: " << model_path
              << " (~" << (model_mem * 3 / 1024 / 1024) << "MB)"
              << " (total loaded: " << sessions_.size()
              << ", total memory: " << (total_memory_ / 1024 / 1024) << "MB)" << std::endl;

    return session;
}

void SessionManager::release(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto ref_it = ref_counts_.find(model_path);
    if (ref_it == ref_counts_.end() || ref_it->second <= 0) {
        return;
    }

    ref_it->second--;
    if (ref_it->second == 0) {
        // Subtract tracked memory before erasing
        auto& session = sessions_[model_path];
        size_t model_mem = 0;
        for (int32_t i = 0; i < session->input_count(); ++i) {
            auto shape = session->get_input_shape(static_cast<size_t>(i));
            size_t elems = 1;
            for (auto d : shape) elems *= static_cast<size_t>(d);
            model_mem += elems * sizeof(float);
        }
        for (int32_t i = 0; i < session->output_count(); ++i) {
            auto shape = session->get_output_shape(static_cast<size_t>(i));
            size_t elems = 1;
            for (auto d : shape) elems *= static_cast<size_t>(d);
            model_mem += elems * sizeof(float);
        }
        total_memory_ -= model_mem * 3;

        sessions_.erase(model_path);
        ref_counts_.erase(ref_it);
        std::cout << "[SessionManager] Unloaded model: " << model_path
                  << " (remaining: " << sessions_.size()
                  << ", total memory: " << (total_memory_ / 1024 / 1024) << "MB)" << std::endl;
    }
}

bool SessionManager::is_loaded(const std::string& model_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.find(model_path) != sessions_.end();
}

size_t SessionManager::total_tpu_memory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_memory_;
}

int SessionManager::loaded_model_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(sessions_.size());
}

}  // namespace inference

#endif  // USE_CVI_TPU
