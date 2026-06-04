// Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace jax_metallib {

class CompletionEventState {
public:
    CompletionEventState() = default;

    bool IsReady() const {
        return ready_.load(std::memory_order_acquire);
    }

    std::string Error() const {
        std::scoped_lock lock(mu_);
        return error_;
    }

    void Await() const {
        // Fast path: already complete.
        if (ready_.load(std::memory_order_acquire)) {
            return;
        }
        // Spin for up to ~200us to avoid condvar wakeup latency (~15-20us).
        // For typical GPU fences (100-200us), spinning catches completion
        // without the kernel-level wakeup overhead of condition variables.
        for (int i = 0; i < 100000; ++i) {
            if (ready_.load(std::memory_order_acquire)) {
                return;
            }
#if defined(__aarch64__) || defined(__arm64__)
            __asm__ volatile("yield");
#elif defined(__x86_64__)
            __asm__ volatile("pause");
#endif
        }
        // Slow path: condvar wait with timeout.
        const auto timeout_ms = []() -> int {
            const char* env = std::getenv("JAX_METALLIB_COMPLETION_TIMEOUT_MS");
            if (env) {
                try {
                    int val = std::stoi(env);
                    return val > 0 ? val : 30000;
                } catch (...) {
                    return 30000;
                }
            }
            return 30000;
        }();

        std::unique_lock lock(mu_);
        bool ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                  [&] { return ready_.load(std::memory_order_relaxed); });
        if (!ready) {
            throw std::runtime_error("CompletionEvent::Await() timed out after " +
                                     std::to_string(timeout_ms) + "ms");
        }
    }

    void MarkReady() {
        FinishWithError("");
    }

    void MarkError(std::string error) {
        FinishWithError(std::move(error));
    }

    // Reset a completed event for reuse (avoids heap allocation).
    // Returns true only if the event was ready (safe to reuse).
    bool TryReset() {
        // Fast check without mutex — avoids lock overhead when not ready.
        if (!ready_.load(std::memory_order_acquire)) {
            return false;
        }
        std::scoped_lock lock(mu_);
        if (!ready_.load(std::memory_order_relaxed)) {
            return false;
        }
        ready_.store(false, std::memory_order_relaxed);
        error_.clear();
        callbacks_.clear();
        return true;
    }

    void AddReadyCallback(std::function<void()> cb) {
        if (!cb) {
            return;
        }
        bool run_now = false;
        {
            std::scoped_lock lock(mu_);
            if (ready_.load(std::memory_order_relaxed)) {
                run_now = true;
            } else {
                callbacks_.push_back(std::move(cb));
            }
        }
        if (run_now) {
            cb();
        }
    }

private:
    void FinishWithError(std::string error) {
        std::vector<std::function<void()>> callbacks;
        {
            std::scoped_lock lock(mu_);
            if (ready_.load(std::memory_order_relaxed)) {
                return;
            }
            ready_.store(true, std::memory_order_release);
            error_ = std::move(error);
            callbacks.swap(callbacks_);
        }
        cv_.notify_all();
        for (auto& cb : callbacks) {
            if (cb) {
                cb();
            }
        }
    }

    mutable std::mutex mu_;
    mutable std::condition_variable cv_;
    std::atomic<bool> ready_{false};
    std::string error_;
    std::vector<std::function<void()>> callbacks_;
};

using CompletionEvent = std::shared_ptr<CompletionEventState>;

inline CompletionEvent MakePendingCompletionEvent() {
    return std::make_shared<CompletionEventState>();
}

inline CompletionEvent MakeReadyCompletionEvent() {

    static CompletionEvent singleton = [] {
        auto e = std::make_shared<CompletionEventState>();
        e->MarkReady();
        return e;
    }();
    return singleton;
}

inline bool IsCompletionEventReady(const CompletionEvent& event) {
    return !event || event->IsReady();
}

inline void AwaitCompletionEvent(const CompletionEvent& event) {
    if (event) {
        event->Await();
    }
}

inline std::string CompletionEventError(const CompletionEvent& event) {
    return event ? event->Error() : "";
}

inline void MarkCompletionEventReady(const CompletionEvent& event) {
    if (event) {
        event->MarkReady();
    }
}

inline void MarkCompletionEventError(const CompletionEvent& event, std::string error) {
    if (event) {
        event->MarkError(std::move(error));
    }
}

} // namespace jax_metallib
