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

    // Waits until the event is ready, however long the GPU work takes. Every wait
    // that guards freeing or reusing a buffer must use this: giving up early would
    // release memory the GPU is still writing. It cannot hang on a stuck GPU,
    // because Metal's command-buffer watchdog completes such a buffer with an
    // error, which fires the completion handler.
    void Await() const {
        AwaitWithHint(0);
    }

    void AwaitWithHint(uint32_t expected_us) const {
        if (SpinUntilReady(expected_us)) {
            return;
        }
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return ready_.load(std::memory_order_relaxed); });
    }

    // Caller-facing wait bounded by JAX_METALLIB_COMPLETION_TIMEOUT_MS (default
    // 30000, read per wait). Never throws: on timeout it returns false and
    // describes the timeout in *error. The event stays pending and the GPU work
    // keeps running, so release waits still wait for it. (Await used to throw
    // std::runtime_error, which reached std::terminate on the detached
    // cache-release thread and across the PJRT C boundary.)
    bool AwaitWithTimeout(std::string* error = nullptr) const {
        if (SpinUntilReady(0)) {
            return true;
        }
        const int timeout_ms = CompletionTimeoutMs();
        std::unique_lock lock(mu_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [&] { return ready_.load(std::memory_order_relaxed); })) {
            return true;
        }
        if (error) {
            *error = "CompletionEvent::Await() timed out after " + std::to_string(timeout_ms) +
                     "ms (raise JAX_METALLIB_COMPLETION_TIMEOUT_MS for longer GPU programs)";
        }
        return false;
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
    // Spins briefly so fast GPU completions skip the condvar; true once ready.
    bool SpinUntilReady(uint32_t expected_us) const {
        if (ready_.load(std::memory_order_acquire)) {
            return true;
        }
        // Phase 1: short spin (5K iterations ~10us on ARM).
        for (int i = 0; i < 5000; ++i) {
            if (ready_.load(std::memory_order_acquire)) {
                return true;
            }
            SpinPause();
        }
        // Phase 2: medium spin (up to JAX_METALLIB_SPIN_ITERATIONS, default 50K).
        // Skip if the hint suggests GPU work is expected to take longer than ~200us.
        if (expected_us <= 200) {
            int max_spin = GetMaxSpinIterations();
            for (int i = 5000; i < max_spin; ++i) {
                if (ready_.load(std::memory_order_acquire)) {
                    return true;
                }
                SpinPause();
            }
        }
        return ready_.load(std::memory_order_acquire);
    }

    static int CompletionTimeoutMs() {
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
    }

    void SpinPause() const {
#if defined(__aarch64__) || defined(__arm64__)
        __asm__ volatile("yield");
#elif defined(__x86_64__)
        __asm__ volatile("pause");
#endif
    }

    static int GetMaxSpinIterations() {
        const char* env = std::getenv("JAX_METALLIB_SPIN_ITERATIONS");
        if (env) {
            try {
                int val = std::stoi(env);
                return val > 0 ? val : 50000;
            } catch (...) {
                return 50000;
            }
        }
        return 50000;
    }

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

// Caller-facing bounded wait; see CompletionEventState::AwaitWithTimeout.
inline bool AwaitCompletionEventWithTimeout(const CompletionEvent& event, std::string* error) {
    return !event || event->AwaitWithTimeout(error);
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
