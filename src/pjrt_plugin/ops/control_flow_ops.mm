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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#import "pjrt_plugin/core/issue_url.h"
#import "pjrt_plugin/ops/registry.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <sstream>

namespace jax_metallib {

namespace {

std::string TypeToString(mlir::Type type) {
    std::string out;
    llvm::raw_string_ostream os(out);
    type.print(os);
    return os.str();
}

std::string DescribeTypeForError(mlir::Type type) {
    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
        std::ostringstream oss;
        oss << "shape=[";
        auto shape = ranked.getShape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            if (shape[i] < 0) {
                oss << "?";
            } else {
                oss << shape[i];
            }
        }
        oss << "], dtype=" << TypeToString(ranked.getElementType());
        return oss.str();
    }
    return TypeToString(type);
}

bool ReadConstantIntScalar(mlir::Value value, int64_t* out) {
    if (!out) {
        return false;
    }
    if (!value) {
        return false;
    }
    auto* defOp = value.getDefiningOp();
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
    if (!constOp) {
        return false;
    }
    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!denseAttr || denseAttr.getNumElements() != 1) {
        return false;
    }
    auto elemType = denseAttr.getElementType();
    auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType);
    if (!intType) {
        return false;
    }
    llvm::APInt v = *denseAttr.getValues<llvm::APInt>().begin();
    *out = intType.isUnsigned() ? (int64_t)v.getZExtValue() : (int64_t)v.getSExtValue();
    return true;
}

static bool EnvFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }
    return !(v[0] == '0' && v[1] == '\0');
}

void BindBlockArguments(mlir::Block& block, NSArray<MPSGraphTensor*>* tensors, ValueMap& values,
                        ValueMap* variables) {
    for (NSUInteger i = 0; i < tensors.count && i < block.getNumArguments(); ++i) {
        void* key = block.getArgument(i).getAsOpaquePointer();
        values[key] = tensors[i];
        if (variables) {
            auto it = variables->find(key);
            if (it != variables->end()) {

                (*variables)[key] = tensors[i];
            }
        }
    }
}

MPSGraphTensor* EvaluateWhileCond(MPSGraph* graph, mlir::Block& condBlock, const ValueMap& values,
                                  NSArray<MPSGraphTensor*>* inputTensors,
                                  NSMutableArray<MPSGraphTensor*>* resultTensors, mlir::ModuleOp module,
                                  int depth, std::string* blockError, BlockProcessor processBlock,
                                  ValueMap* variables) {
    ValueMap condValues = values;
    BindBlockArguments(condBlock, inputTensors, condValues, variables);

    HandlerContext condCtx(graph, nullptr, condValues, module, depth + 1, processBlock);
    condCtx.variables = variables;
    condCtx.in_control_flow = true;
    ScopedControlFlowContext cf_guard;
    ProcessResult condResult = processBlock(condCtx, condBlock);
    if (!condResult.ok() || condResult.return_values.empty()) {
        *blockError = condResult.ok() ? "while cond returned no predicate" : condResult.error;
        [resultTensors addObjectsFromArray:inputTensors];
        return [graph constantWithScalar:0 dataType:MPSDataTypeBool];
    }

    [resultTensors addObjectsFromArray:inputTensors];
    MPSGraphTensor* pred = GetTensor(condValues, condResult.return_values[0]);
    if (!pred) {
        *blockError = "while cond predicate tensor not found";
        return [graph constantWithScalar:0 dataType:MPSDataTypeBool];
    }
    return pred;
}

NSArray<MPSGraphTensor*>* EvaluateLoopBody(MPSGraph* graph, mlir::Block& bodyBlock, const ValueMap& values,
                                           NSArray<MPSGraphTensor*>* bodyArgs, mlir::ModuleOp module,
                                           int depth, bool markWhileContext, std::string* blockError,
                                           BlockProcessor processBlock, ValueMap* variables) {
    ValueMap bodyValues = values;
    BindBlockArguments(bodyBlock, bodyArgs, bodyValues, variables);

    HandlerContext bodyCtx(graph, nullptr, bodyValues, module, depth + 1, processBlock);
    bodyCtx.variables = variables;
    if (markWhileContext) {
        bodyCtx.in_control_flow = true;
    }
    ProcessResult bodyResult;
    if (markWhileContext) {
        ScopedControlFlowContext cf_guard;
        bodyResult = processBlock(bodyCtx, bodyBlock);
    } else {
        bodyResult = processBlock(bodyCtx, bodyBlock);
    }
    if (!bodyResult.ok()) {
        *blockError = bodyResult.error;
        return bodyArgs;
    }

    NSMutableArray<MPSGraphTensor*>* out = [NSMutableArray array];
    for (mlir::Value value : bodyResult.return_values) {
        MPSGraphTensor* tensor = GetTensor(bodyValues, value);
        if (!tensor) {
            *blockError = "while body return tensor not found";
            return bodyArgs;
        }
        [out addObject:tensor];
    }
    return out;
}

struct CountedWhile {
    unsigned counter_index = 0;
    bool upper_bound_is_const = true;
    int64_t upper_bound_const = 0;
    unsigned upper_bound_index = 0;
    int64_t step = 1;
};

static size_t FloatElemByteSize(mlir::Type elemType) {
    if (elemType.isF16() || elemType.isBF16()) {
        return 2;
    }
    if (elemType.isF32()) {
        return 4;
    }
    if (elemType.isF64()) {
        return 8;
    }
    return 0;
}

static size_t StaticTensorByteSizeIfFloat(mlir::Type type) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!ranked || !ranked.hasStaticShape()) {
        return 0;
    }
    size_t elemSize = FloatElemByteSize(ranked.getElementType());
    if (elemSize == 0) {
        return 0;
    }
    size_t bytes = elemSize;
    for (int64_t dim : ranked.getShape()) {
        if (dim <= 0) {
            return 0;
        }
        bytes *= static_cast<size_t>(dim);
    }
    return bytes;
}

static void CollectCarriedIndicesUpdatedByDynamicUpdateSlice(mlir::stablehlo::WhileOp whileOp,
                                                             mlir::Block& bodyBlock, mlir::ModuleOp module,
                                                             std::unordered_set<unsigned>* out) {
    if (!out) {
        return;
    }

    bodyBlock.walk([&](mlir::stablehlo::DynamicUpdateSliceOp dus) {
        mlir::Value base = dus->getOperand(0);
        auto ba = mlir::dyn_cast<mlir::BlockArgument>(base);
        if (!ba || ba.getOwner() != &bodyBlock) {
            return;
        }
        out->insert(ba.getArgNumber());
    });

    bodyBlock.walk([&](mlir::func::CallOp callOp) {
        if (!module) {
            return;
        }
        auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
        if (!callee || callee.empty()) {
            return;
        }

        std::unordered_set<unsigned> updatedArgs;
        callee.walk([&](mlir::stablehlo::DynamicUpdateSliceOp dus) {
            mlir::Value base = dus->getOperand(0);
            auto ba = mlir::dyn_cast<mlir::BlockArgument>(base);
            if (!ba || ba.getOwner() != &callee.front()) {
                return;
            }
            updatedArgs.insert(ba.getArgNumber());
        });

        if (updatedArgs.empty()) {
            return;
        }

        for (unsigned argIndex : updatedArgs) {
            if (argIndex >= callOp.getNumOperands()) {
                continue;
            }
            mlir::Value actual = callOp.getOperand(argIndex);
            auto ba = mlir::dyn_cast<mlir::BlockArgument>(actual);
            if (!ba || ba.getOwner() != &bodyBlock) {
                continue;
            }
            out->insert(ba.getArgNumber());
        }
    });
}

static bool MatchCountedWhile(mlir::stablehlo::WhileOp whileOp, mlir::Block& condBlock,
                              mlir::Block& bodyBlock, CountedWhile* out) {
    if (!out) {
        return false;
    }
    if (condBlock.getNumArguments() != bodyBlock.getNumArguments()) {
        return false;
    }
    auto* condTerm = condBlock.getTerminator();
    auto condRet = mlir::dyn_cast_or_null<mlir::stablehlo::ReturnOp>(condTerm);
    if (!condRet || condRet->getNumOperands() != 1) {
        return false;
    }

    mlir::Value pred = condRet->getOperand(0);
    auto cmp = pred.getDefiningOp<mlir::stablehlo::CompareOp>();
    if (!cmp) {
        return false;
    }
    using Dir = mlir::stablehlo::ComparisonDirection;
    if (cmp.getComparisonDirection() != Dir::LT) {
        return false;
    }

    mlir::Value lhs = cmp.getLhs();
    mlir::Value rhs = cmp.getRhs();
    auto counterArg = mlir::dyn_cast<mlir::BlockArgument>(lhs);
    mlir::Value upperVal = rhs;
    if (!counterArg || counterArg.getOwner() != &condBlock) {
        return false;
    }

    bool upperIsConst = true;
    int64_t upperConst = 0;
    unsigned upperIndex = 0;
    if (!ReadConstantIntScalar(upperVal, &upperConst)) {
        auto upperArg = mlir::dyn_cast<mlir::BlockArgument>(upperVal);
        if (!upperArg || upperArg.getOwner() != &condBlock) {
            return false;
        }
        upperIsConst = false;
        upperIndex = upperArg.getArgNumber();
    }

    unsigned counterIndex = counterArg.getArgNumber();
    if (counterIndex >= bodyBlock.getNumArguments()) {
        return false;
    }
    if (counterIndex >= whileOp->getNumOperands() || counterIndex >= whileOp->getNumResults()) {
        return false;
    }

    auto* bodyTerm = bodyBlock.getTerminator();
    auto bodyRet = mlir::dyn_cast_or_null<mlir::stablehlo::ReturnOp>(bodyTerm);
    if (!bodyRet || bodyRet->getNumOperands() != whileOp->getNumOperands()) {
        return false;
    }
    if (!upperIsConst) {
        if (upperIndex >= bodyBlock.getNumArguments()) {
            return false;
        }
        if (upperIndex >= whileOp->getNumOperands() || upperIndex >= whileOp->getNumResults()) {
            return false;
        }

        if (bodyRet->getOperand(upperIndex) != bodyBlock.getArgument(upperIndex)) {
            return false;
        }
    }

    mlir::Value nextCounter = bodyRet->getOperand(counterIndex);
    auto add = nextCounter.getDefiningOp<mlir::stablehlo::AddOp>();
    if (!add) {
        return false;
    }

    mlir::Value addLhs = add.getLhs();
    mlir::Value addRhs = add.getRhs();
    mlir::BlockArgument bodyCounterArg = bodyBlock.getArgument(counterIndex);
    mlir::Value stepVal;
    if (addLhs == bodyCounterArg) {
        stepVal = addRhs;
    } else if (addRhs == bodyCounterArg) {
        stepVal = addLhs;
    } else {
        return false;
    }

    int64_t stepConst = 0;
    if (!ReadConstantIntScalar(stepVal, &stepConst) || stepConst <= 0) {
        return false;
    }

    out->counter_index = counterIndex;
    out->upper_bound_is_const = upperIsConst;
    out->upper_bound_const = upperConst;
    out->upper_bound_index = upperIndex;
    out->step = stepConst;
    return true;
}

} // namespace

static int max_while_depth = [] {
    const char* env = std::getenv("JAX_METALLIB_MAX_WHILE_DEPTH");
    return env ? std::stoi(env) : 100;
}();

static ProcessResult HandleWhileOp(HandlerContext& ctx) {
    auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(ctx.op);
    if (!whileOp) {
        return ProcessResult::Error("Expected stablehlo.while operation");
    }

    if (ctx.depth > max_while_depth) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive while");
    }

    NSMutableArray<MPSGraphTensor*>* initialInputs = [NSMutableArray array];
    for (mlir::Value operand : whileOp->getOperands()) {
        MPSGraphTensor* t = GetTensor(ctx.values, operand);
        if (!t)
            return ProcessResult::Error("While operand tensor not found");
        [initialInputs addObject:t];
    }

    if (whileOp.getCond().empty() || whileOp.getBody().empty()) {
        return ProcessResult::Error("stablehlo.while requires non-empty cond/body regions");
    }
    mlir::Block& condBlock = whileOp.getCond().front();
    mlir::Block& bodyBlock = whileOp.getBody().front();

    MPSGraph* graph = ctx.graph;
    ValueMap& values = ctx.values;
    mlir::ModuleOp module = ctx.module;
    int depth = ctx.depth;
    BlockProcessor processBlock = ctx.processBlock;

    __block std::string blockError;

    CountedWhile counted;
    if (MatchCountedWhile(whileOp, condBlock, bodyBlock, &counted)) {
        const bool enable_variable_carry = !EnvFlagEnabled("JAX_METALLIB_DISABLE_VARIABLE_CARRY");

        std::unordered_set<unsigned> promoted;
        CollectCarriedIndicesUpdatedByDynamicUpdateSlice(whileOp, bodyBlock, module, &promoted);

        ValueMap variable_map;
        ValueMap* variables = nullptr;
        if (enable_variable_carry && !promoted.empty()) {
            if (@available(macOS 15.0, *)) {
                for (unsigned idx : promoted) {
                    if (idx >= initialInputs.count) {
                        continue;
                    }

                    size_t bytes = StaticTensorByteSizeIfFloat(whileOp->getOperand(idx).getType());
                    if (bytes < (64 * 1024)) {
                        continue;
                    }
                    MPSGraphTensor* init = initialInputs[idx];
                    if (!init) {
                        continue;
                    }
                    MPSGraphTensor* var = [graph variableFromTensorWithTensor:init name:nil];
                    if (!var) {
                        continue;
                    }
                    initialInputs[idx] = var;
                    if (idx < condBlock.getNumArguments()) {
                        variable_map[condBlock.getArgument(idx).getAsOpaquePointer()] = var;
                    }
                    if (idx < bodyBlock.getNumArguments()) {
                        variable_map[bodyBlock.getArgument(idx).getAsOpaquePointer()] = var;
                    }
                }
                if (!variable_map.empty()) {
                    variables = &variable_map;
                }
            }
        }

        MPSGraphTensor* lowerBound = initialInputs[counted.counter_index];
        if (!lowerBound) {
            return ProcessResult::Error("stablehlo.while: missing lowerBound tensor");
        }
        MPSGraphTensor* upperBound = nil;
        if (counted.upper_bound_is_const) {
            upperBound = [graph constantWithScalar:counted.upper_bound_const dataType:lowerBound.dataType];
        } else {
            if (counted.upper_bound_index >= initialInputs.count) {
                return ProcessResult::Error("stablehlo.while: invalid dynamic upperBound index");
            }
            upperBound = initialInputs[counted.upper_bound_index];
            if (!upperBound) {
                return ProcessResult::Error("stablehlo.while: missing dynamic upperBound tensor");
            }
            if (upperBound.dataType != lowerBound.dataType) {
                upperBound = [graph castTensor:upperBound toType:lowerBound.dataType name:nil];
            }
        }
        MPSGraphTensor* step = [graph constantWithScalar:counted.step dataType:lowerBound.dataType];

        NSArray<MPSGraphTensor*>* outputs = [graph
            forLoopWithLowerBound:lowerBound
                       upperBound:upperBound
                             step:step
             initialBodyArguments:initialInputs
                             body:^NSArray<MPSGraphTensor*>*(MPSGraphTensor*,
                                                             NSArray<MPSGraphTensor*>* iterationArguments) {
                               return EvaluateLoopBody(graph, bodyBlock, values, iterationArguments, module,
                                                       depth, true, &blockError, processBlock, variables);
                             }
                             name:nil];

        if (!blockError.empty())
            return ProcessResult::Error(blockError);
        if (!outputs)
            return ProcessResult::Error("forLoopWithLowerBound returned null");
        if ((NSUInteger)whileOp->getNumResults() != outputs.count) {
            return ProcessResult::Error("while output arity mismatch");
        }

        for (NSUInteger i = 0; i < outputs.count; ++i) {

            MPSGraphTensor* outTensor = outputs[i];
            if (variables && variable_map.find(bodyBlock.getArgument((unsigned)i).getAsOpaquePointer()) !=
                                 variable_map.end()) {
                outTensor = [graph readVariable:outputs[i] name:nil];
            }
            ctx.values[whileOp->getResult(i).getAsOpaquePointer()] = outTensor;
        }
        return ProcessResult{};
    }

    NSArray<MPSGraphTensor*>* outputs = [graph whileWithInitialInputs:initialInputs
        before:^MPSGraphTensor*(NSArray<MPSGraphTensor*>* inputTensors,
                                NSMutableArray<MPSGraphTensor*>* resultTensors) {
          return EvaluateWhileCond(graph, condBlock, values, inputTensors, resultTensors, module, depth,
                                   &blockError, processBlock, nullptr);
        }
        after:^NSArray<MPSGraphTensor*>*(NSArray<MPSGraphTensor*>* bodyArgs) {
          return EvaluateLoopBody(graph, bodyBlock, values, bodyArgs, module, depth, true, &blockError,
                                  processBlock, nullptr);
        }
        name:nil];

    if (!blockError.empty())
        return ProcessResult::Error(blockError);
    if (!outputs)
        return ProcessResult::Error("whileWithInitialInputs returned null");
    if ((NSUInteger)whileOp->getNumResults() != outputs.count) {
        return ProcessResult::Error("while output arity mismatch");
    }

    for (NSUInteger i = 0; i < outputs.count; ++i) {
        ctx.values[whileOp->getResult(i).getAsOpaquePointer()] = outputs[i];
    }
    return ProcessResult{};
}

static ProcessResult HandleCaseOp(HandlerContext& ctx) {
    if (ctx.depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive case");
    }
    if (ctx.op->getNumOperands() < 1) {
        return ProcessResult::Error("stablehlo.case requires selector operand");
    }
    if (ctx.op->getNumRegions() < 1) {
        return ProcessResult::Error("stablehlo.case requires at least one branch region");
    }

    MPSGraphTensor* selector = GetTensor(ctx.values, ctx.op->getOperand(0));
    if (!selector) {
        return ProcessResult::Error("stablehlo.case selector tensor not found");
    }

    const size_t numResults = ctx.op->getNumResults();
    const size_t numBranches = ctx.op->getNumRegions();
    const size_t numBranchOperands = ctx.op->getNumOperands() - 1;

    auto evaluateBranch = [&](size_t branchIndex, std::string* error) -> NSArray<MPSGraphTensor*>* {
        if (branchIndex >= numBranches) {
            if (error) {
                *error = "stablehlo.case invalid branch index";
            }
            return nil;
        }

        mlir::Region& region = ctx.op->getRegion((unsigned)branchIndex);
        if (region.empty()) {
            if (error) {
                *error = "stablehlo.case branch region is empty";
            }
            return nil;
        }

        mlir::Block& block = region.front();
        if (block.getNumArguments() > numBranchOperands) {
            if (error) {
                *error = "stablehlo.case branch expects more operands than provided";
            }
            return nil;
        }

        ValueMap branchValues = ctx.values;
        ValueMap branchVariables;
        if (ctx.variables) {
            branchVariables = *ctx.variables;
        }
        for (size_t i = 0; i < block.getNumArguments(); ++i) {
            mlir::Value branchOperand = ctx.op->getOperand(1 + i);
            MPSGraphTensor* argTensor = GetTensor(ctx.values, branchOperand);
            if (!argTensor) {
                if (error) {
                    *error = "stablehlo.case branch operand tensor not found";
                }
                return nil;
            }
            void* key = block.getArgument((unsigned)i).getAsOpaquePointer();
            branchValues[key] = argTensor;
            if (ctx.variables) {
                auto it = ctx.variables->find(branchOperand.getAsOpaquePointer());
                if (it != ctx.variables->end()) {
                    branchVariables[key] = it->second;
                }
            }
        }

        HandlerContext branchCtx(ctx.graph, nullptr, branchValues, ctx.module, ctx.depth + 1,
                                 ctx.processBlock);
        branchCtx.variables = ctx.variables ? &branchVariables : nullptr;
        ProcessResult branchResult = ctx.processBlock(branchCtx, block);
        if (!branchResult.ok()) {
            if (error) {
                *error = branchResult.error;
            }
            return nil;
        }
        if (branchResult.return_values.size() != numResults) {
            if (error) {
                *error = "stablehlo.case branch result arity mismatch";
            }
            return nil;
        }

        NSMutableArray<MPSGraphTensor*>* out = [NSMutableArray arrayWithCapacity:(NSUInteger)numResults];
        for (size_t r = 0; r < numResults; ++r) {
            MPSGraphTensor* t = GetTensor(branchValues, branchResult.return_values[r]);
            if (!t) {
                if (error) {
                    *error = "stablehlo.case branch return tensor not found";
                }
                return nil;
            }
            [out addObject:t];
        }
        return out;
    };

    std::vector<NSArray<MPSGraphTensor*>*> branchOutputs(numBranches);
    for (size_t b = 0; b < numBranches; ++b) {
        std::string error;
        NSArray<MPSGraphTensor*>* out = evaluateBranch(b, &error);
        if (!out) {
            return ProcessResult::Error(error);
        }
        branchOutputs[b] = out;
    }

    for (size_t r = 0; r < numResults; ++r) {
        MPSGraphTensor* selected = branchOutputs[numBranches - 1][r];
        for (size_t i = numBranches - 1; i > 0; --i) {
            MPSGraphTensor* branchIndex = [ctx.graph constantWithScalar:static_cast<double>(i - 1)
                                                               dataType:selector.dataType];
            MPSGraphTensor* pred = [ctx.graph equalWithPrimaryTensor:selector
                                                     secondaryTensor:branchIndex
                                                                name:nil];
            selected = [ctx.graph selectWithPredicateTensor:pred
                                        truePredicateTensor:branchOutputs[i - 1][r]
                                       falsePredicateTensor:selected
                                                       name:nil];
        }
        ctx.values[ctx.op->getResult((unsigned)r).getAsOpaquePointer()] = selected;
    }

    return ProcessResult{};
}

static ProcessResult HandleCustomCall(HandlerContext& ctx) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(ctx.op);
    if (!customCallOp) {
        return ProcessResult::Error("custom_call: expected CustomCallOp");
    }

    std::string target = customCallOp.getCallTargetName().str();

    const OpHandler* handler = CustomCallRegistry::Find(target);
    if (!handler) {
        std::string op_name = "stablehlo.custom_call(" + target + ")";

        std::ostringstream oss;
        oss << "Unsupported custom call target '" << target << "'.\n";
        oss << "Operands (" << ctx.op->getNumOperands() << "):\n";
        for (unsigned i = 0; i < ctx.op->getNumOperands(); ++i) {
            oss << "  operand[" << i << "]: " << DescribeTypeForError(ctx.op->getOperand(i).getType())
                << "\n";
        }
        oss << "Results (" << ctx.op->getNumResults() << "):\n";
        for (unsigned i = 0; i < ctx.op->getNumResults(); ++i) {
            oss << "  result[" << i << "]: " << DescribeTypeForError(ctx.op->getResult(i).getType()) << "\n";
        }
        oss << "\n" << UnsupportedOpsMessage({op_name});
        return ProcessResult::Error(oss.str());
    }

    return handler->graph_handler(ctx);
}

REGISTER_MPS_OP("stablehlo.while", HandleWhileOp);
REGISTER_MPS_OP("stablehlo.case", HandleCaseOp);

REGISTER_MPS_OP("stablehlo.custom_call", HandleCustomCall);

} // namespace jax_metallib
