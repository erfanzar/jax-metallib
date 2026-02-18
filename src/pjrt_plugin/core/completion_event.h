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

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace jax_metallib {

class CompletionEventState {
public:
    CompletionEventState() = default;

    bool IsReady() const {
        std::scoped_lock lock(mu_);
        return ready_;
    }

    std::string Error() const {
        std::scoped_lock lock(mu_);
        return error_;
    }

    void Await() const {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return ready_; });
    }

    void MarkReady() {
        FinishWithError("");
    }

    void MarkError(std::string error) {
        FinishWithError(std::move(error));
    }

    void AddReadyCallback(std::function<void()> cb) {
        if (!cb) {
            return;
        }
        bool run_now = false;
        {
            std::scoped_lock lock(mu_);
            if (ready_) {
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
            if (ready_) {
                return;
            }
            ready_ = true;
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
    bool ready_ = false;
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
