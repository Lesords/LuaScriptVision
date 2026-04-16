#pragma once

#ifdef USE_CVI_TPU

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace inference {

class CviSession;

// Manages TPU runtime handle (global singleton) and CviSession lifecycle.
// - CVI_RT_Init called once on first acquire()
// - CVI_RT_DeInit called on destruction (process exit)
// - Same model_path returns the same CviSession (reference counted)
class SessionManager {
public:
    static SessionManager& instance();

    // Load or retrieve a cached session. Reference count +1.
    std::shared_ptr<CviSession> acquire(const std::string& model_path);

    // Release a session. Reference count -1. Unloads when count reaches 0.
    void release(const std::string& model_path);

    // Check if a model is already loaded (no memory cost to acquire again)
    bool is_loaded(const std::string& model_path) const;

    size_t total_tpu_memory() const;
    int loaded_model_count() const;

private:
    SessionManager();
    ~SessionManager();

    SessionManager(const SessionManager&) = delete;
    SessionManager& operator=(const SessionManager&) = delete;

    void* rt_handle_ = nullptr;  // CVI_RT_HANDLE, typed in .cpp
    bool rt_initialized_ = false;
    std::map<std::string, std::shared_ptr<CviSession>> sessions_;
    std::map<std::string, int> ref_counts_;
    size_t total_memory_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace inference

#endif  // USE_CVI_TPU
