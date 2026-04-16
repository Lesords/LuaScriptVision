#pragma once

#ifdef USE_CVI_TPU

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace inference {

class CviSession;

struct TpuResult {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> output_shapes;
    double input_ms = 0.0;
    double forward_ms = 0.0;
    double output_ms = 0.0;
    bool success = false;
    std::string error;
};

// Single-threaded TPU inference scheduler.
// Serializes all CVI_NN_Forward calls so multiple ModelNodes can
// safely share the TPU via frame-level round-robin.
class TpuScheduler {
public:
    static TpuScheduler& instance();

    // Submit a VB-memory inference task. Blocks until result is ready.
    TpuResult submit_vb(CviSession* session, uint64_t input_phys_addr, size_t input_size);

    // Submit a float-data inference task. Blocks until result is ready.
    TpuResult submit_float(CviSession* session,
                           const float* input_data, size_t input_elements,
                           const std::vector<int64_t>& input_shape);

    void start();
    void stop();

private:
    TpuScheduler();
    ~TpuScheduler();

    TpuScheduler(const TpuScheduler&) = delete;
    TpuScheduler& operator=(const TpuScheduler&) = delete;

    struct Task {
        CviSession* session;
        enum Mode { VB, FLOAT } mode;
        // VB mode
        uint64_t vb_phys_addr = 0;
        size_t vb_size = 0;
        // Float mode
        std::vector<float> float_data;
        std::vector<int64_t> float_shape;
        // Result
        std::promise<TpuResult> promise;
    };

    void runLoop();

    std::thread worker_;
    std::deque<std::unique_ptr<Task>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    std::atomic<bool> started_{false};
};

}  // namespace inference

#endif  // USE_CVI_TPU
