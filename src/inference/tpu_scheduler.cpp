#include "tpu_scheduler.h"
#include "cvi_session.h"

#ifdef USE_CVI_TPU

#include <iostream>

namespace inference {

TpuScheduler& TpuScheduler::instance() {
    static TpuScheduler inst;
    return inst;
}

TpuScheduler::TpuScheduler() = default;

TpuScheduler::~TpuScheduler() {
    stop();
}

void TpuScheduler::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) {
        return;  // Already started
    }

    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&TpuScheduler::runLoop, this);
}

void TpuScheduler::stop() {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();

    if (worker_.joinable()) {
        worker_.join();
    }
    started_.store(false, std::memory_order_release);
}

TpuResult TpuScheduler::submit_vb(CviSession* session,
                                   uint64_t input_phys_addr,
                                   size_t input_size) {
    auto task = std::make_unique<Task>();
    task->session = session;
    task->mode = Task::VB;
    task->vb_phys_addr = input_phys_addr;
    task->vb_size = input_size;

    std::future<TpuResult> future = task->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lazy start worker on first submission
        if (!started_.load(std::memory_order_acquire)) {
            start();
        }
        queue_.push_back(std::move(task));
    }
    cv_.notify_one();

    return future.get();
}

TpuResult TpuScheduler::submit_float(CviSession* session,
                                      const float* input_data,
                                      size_t input_elements,
                                      const std::vector<int64_t>& input_shape) {
    auto task = std::make_unique<Task>();
    task->session = session;
    task->mode = Task::FLOAT;
    task->float_data.assign(input_data, input_data + input_elements);
    task->float_shape = input_shape;

    std::future<TpuResult> future = task->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_.load(std::memory_order_acquire)) {
            start();
        }
        queue_.push_back(std::move(task));
    }
    cv_.notify_one();

    return future.get();
}

void TpuScheduler::runLoop() {
    while (running_.load(std::memory_order_acquire)) {
        std::unique_ptr<Task> task;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !queue_.empty() || !running_.load(std::memory_order_acquire);
            });

            if (!running_.load(std::memory_order_acquire) && queue_.empty()) {
                break;
            }

            if (!queue_.empty()) {
                task = std::move(queue_.front());
                queue_.pop_front();
            }
        }

        if (!task) continue;

        TpuResult result;
        try {
            if (task->mode == Task::VB) {
                task->session->run_vb(
                    task->vb_phys_addr, task->vb_size,
                    &result.outputs, &result.output_shapes);
            } else {
                task->session->run_all(
                    task->float_data.data(), task->float_shape,
                    &result.outputs, &result.output_shapes);
            }

            auto stats = task->session->last_run_stats();
            result.input_ms = stats.input_ms;
            result.forward_ms = stats.forward_ms;
            result.output_ms = stats.output_ms;
            result.success = true;
        } catch (const std::exception& e) {
            result.success = false;
            result.error = e.what();
        } catch (...) {
            result.success = false;
            result.error = "Unknown TPU inference error";
        }

        task->promise.set_value(std::move(result));
    }
}

}  // namespace inference

#endif  // USE_CVI_TPU
