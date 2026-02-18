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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "pjrt_plugin/core/completion_event.h"
#include "pjrt_plugin/core/mps_buffer.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mps {
struct ParsedModule;
}

namespace jax_metallib {

class MpsClient;
class MpsDevice;

struct ExecutionResult {
    std::vector<std::unique_ptr<MpsBuffer>> buffers;
    CompletionEvent completion;
    std::string error;

    bool ok() const {
        return error.empty();
    }
    static ExecutionResult Error(const std::string& msg) {
        ExecutionResult r;
        r.error = msg;
        return r;
    }
};

struct ExecutionPlan;

class MpsExecutable {
public:
    MpsExecutable(MpsClient* client, mps::ParsedModule module);

    ~MpsExecutable();

    bool IsValid() const {
        return valid_;
    }

    const std::string& error() const {
        return error_;
    }

    ExecutionResult Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device);

    std::string Serialize() const;

    std::string SerializeToText() const;

    const std::string& name() const {
        return name_;
    }
    int num_outputs() const {
        return num_outputs_;
    }

private:
    bool BuildExecutionPlan();

    MpsClient* client_;
    std::string name_;
    std::string error_;
    int num_outputs_ = 1;
    bool valid_ = false;

    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    mlir::func::FuncOp entry_func_;

    std::mutex plan_mutex_;
    bool plan_built_ = false;
    std::unique_ptr<ExecutionPlan> plan_;
};

} // namespace jax_metallib
