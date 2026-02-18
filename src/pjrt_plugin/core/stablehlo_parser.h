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

#ifndef STABLEHLO_PARSER_H
#define STABLEHLO_PARSER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <memory>
#include <string>
#include <vector>

namespace mps {

struct ParsedModule {
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp entry_func;

    std::vector<std::string> unsupported_ops;

    bool ok() const {
        return context && module && entry_func;
    }
};

ParsedModule parseStableHLOBytecode(const char* data, size_t size);

ParsedModule parseStableHLOText(const std::string& text);

std::string bytecodeToText(const char* data, size_t size);

} // namespace mps

#endif
