#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>

namespace node {

// Thread-safe message queue for inter-node communication
template<typename T>
class MessageBox {
public:
    using Deleter = std::function<void(T*)>;

    explicit MessageBox(size_t capacity = 4, Deleter deleter = nullptr)
        : capacity_(capacity), deleter_(deleter) {}

    ~MessageBox() {
        clear();
    }

    // Non-blocking post (returns false if queue is full)
    bool post(T* msg, uint32_t timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (interrupted_.load(std::memory_order_acquire)) {
            return false;
        }

        if (timeout_ms == 0) {
            // Non-blocking
            if (queue_.size() >= capacity_) {
                return false;
            }
        } else {
            // Wait for space (or interrupt)
            auto predicate = [this]() {
                return queue_.size() < capacity_ || interrupted_.load(std::memory_order_acquire);
            };

            if (timeout_ms == UINT32_MAX) {
                not_full_.wait(lock, predicate);
            } else {
                if (!not_full_.wait_for(lock,
                        std::chrono::milliseconds(timeout_ms), predicate)) {
                    return false;
                }
            }

            // Check if interrupted
            if (interrupted_.load(std::memory_order_acquire)) {
                return false;
            }
        }

        queue_.push(msg);
        not_empty_.notify_one();
        return true;
    }

    // Blocking fetch with timeout (returns false if timeout or interrupted)
    bool fetch(T** msg, uint32_t timeout_ms = UINT32_MAX) {
        std::unique_lock<std::mutex> lock(mutex_);

        auto predicate = [this]() {
            return !queue_.empty() || interrupted_.load(std::memory_order_acquire);
        };

        if (timeout_ms == 0) {
            if (queue_.empty()) {
                return false;
            }
        } else if (timeout_ms == UINT32_MAX) {
            not_empty_.wait(lock, predicate);
        } else {
            if (!not_empty_.wait_for(lock,
                    std::chrono::milliseconds(timeout_ms), predicate)) {
                return false;
            }
        }

        // Check if interrupted (and queue is still empty)
        if (interrupted_.load(std::memory_order_acquire) && queue_.empty()) {
            return false;
        }

        *msg = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    // Status queries
    size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool isEmpty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    bool isFull() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size() >= capacity_;
    }

    size_t capacity() const {
        return capacity_;
    }

    // Clear all messages (calls deleter if set)
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            T* item = queue_.front();
            queue_.pop();
            if (deleter_) {
                deleter_(item);
            } else if constexpr (std::is_destructible_v<T>) {
                delete item;
            }
            // else: T has inaccessible destructor; user must provide a Deleter
        }
    }

    // Interrupt blocking operations (for graceful shutdown)
    void interrupt() {
        interrupted_.store(true, std::memory_order_release);
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    // Reset interrupt flag (for reuse)
    void reset() {
        interrupted_.store(false, std::memory_order_release);
    }

private:
    std::queue<T*> queue_;
    size_t capacity_;
    Deleter deleter_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::atomic<bool> interrupted_{false};
};

} // namespace node
