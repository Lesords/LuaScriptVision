#include "session_manager.h"

#ifdef USE_CVI_TPU

#include "cvi_session.h"

#include <cviruntime.h>
#include <iostream>
#include <stdexcept>

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
        return it->second;
    }

    // Load new session with shared RT handle
    auto rt = static_cast<CVI_RT_HANDLE>(rt_handle_);
    auto session = std::make_shared<CviSession>(model_path, rt);
    sessions_[model_path] = session;
    ref_counts_[model_path] = 1;

    std::cout << "[SessionManager] Loaded model: " << model_path
              << " (total loaded: " << sessions_.size() << ")" << std::endl;

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
        sessions_.erase(model_path);
        ref_counts_.erase(ref_it);
        std::cout << "[SessionManager] Unloaded model: " << model_path
                  << " (remaining: " << sessions_.size() << ")" << std::endl;
    }
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
