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

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <vector>

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
                                           BlockProcessor processBlock, ValueMap* variables,
                                           const std::vector<mlir::Value>* captureValues = nullptr,
                                           std::vector<MPSGraphTensor*>* captured = nullptr) {
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
    if (captureValues && captured) {
        for (mlir::Value v : *captureValues) {
            MPSGraphTensor* tensor = GetTensor(bodyValues, v);
            if (!tensor) {
                *blockError = "while body capture tensor not found";
                return bodyArgs;
            }
            captured->push_back(tensor);
        }
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

// MPSGraph whileWithInitialInputs segfaulted (first run, deterministic) when a
// loop carry read by the predicate, and produced by a reduction in the body,
// was not a compile target: e.g. a convergence loop testing max|a| whose jitted
// function returned only `a` (reproduced in a standalone MPSGraph program;
// making the carry a target operation or a control dependency did not help,
// a data dependency did). Tie every output to one zero scalar that depends on
// one element of each output. float(isNaN(e)) * 0 is exactly 0 even for
// NaN/inf carries, and x - 0 keeps the sign of -0. Complex outputs are left
// untied. JAX_METALLIB_DISABLE_WHILE_OUTPUT_TIE=1 disables this.
static NSArray<MPSGraphTensor*>* TieWhileOutputs(MPSGraph* g, NSArray<MPSGraphTensor*>* outputs) {
    if (outputs.count < 2 || EnvFlagEnabled("JAX_METALLIB_DISABLE_WHILE_OUTPUT_TIE")) {
        return outputs;
    }
    auto isComplex = [](MPSDataType t) {
        return t == MPSDataTypeComplexFloat32 || t == MPSDataTypeComplexFloat16;
    };
    MPSGraphTensor* zeroF = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* z = nil;
    for (MPSGraphTensor* t in outputs) {
        NSArray<NSNumber*>* shape = t.shape;
        if (!shape) {
            continue;
        }
        int64_t numel = 1;
        for (NSNumber* d in shape) {
            numel *= [d longLongValue];
        }
        if (numel <= 0) {
            continue;
        }
        MPSGraphTensor* e = isComplex(t.dataType) ? [g realPartOfTensor:t name:nil] : t;
        e = [g reshapeTensor:e withShape:@[@(numel)] name:nil];
        e = [g sliceTensor:e dimension:0 start:0 length:1 name:nil];
        if (e.dataType != MPSDataTypeFloat32) {
            e = [g castTensor:e toType:MPSDataTypeFloat32 name:nil];
        }
        MPSGraphTensor* nanF = [g castTensor:[g isNaNWithTensor:e name:nil]
                                      toType:MPSDataTypeFloat32
                                        name:nil];
        MPSGraphTensor* zj = [g multiplicationWithPrimaryTensor:nanF secondaryTensor:zeroF name:nil];
        z = z ? [g additionWithPrimaryTensor:z secondaryTensor:zj name:nil] : zj;
    }
    if (!z) {
        return outputs;
    }
    z = [g reshapeTensor:z withShape:@[] name:nil];
    NSMutableArray<MPSGraphTensor*>* tied = [NSMutableArray arrayWithCapacity:outputs.count];
    for (MPSGraphTensor* t in outputs) {
        MPSDataType dt = t.dataType;
        if (isComplex(dt)) {
            [tied addObject:t];
        } else if (dt == MPSDataTypeBool) {
            MPSGraphTensor* zb = [g castTensor:z toType:MPSDataTypeBool name:nil];
            [tied addObject:[g logicalORWithPrimaryTensor:t secondaryTensor:zb name:nil]];
        } else {
            MPSGraphTensor* zt = dt == MPSDataTypeFloat32 ? z : [g castTensor:z toType:dt name:nil];
            [tied addObject:[g subtractionWithPrimaryTensor:t secondaryTensor:zt name:nil]];
        }
    }
    return tied;
}

// Each MPSGraph for-loop iteration costs ~5.4 us regardless of body size
// (1024-step empty scan: 5.8 ms; unrolled by 8: 1.4 ms). For counted loops
// whose trip count is a compile-time constant, evaluate the body `unroll`
// times per MPSGraph iteration. Only small bodies without nested control flow
// or in-loop buffer writes are unrolled, to bound graph size and compile time.
// JAX_METALLIB_LOOP_UNROLL=<k> sets the maximum factor (default 8; 1 disables).
static int MaxLoopUnroll() {
    static const int value = [] {
        const char* v = std::getenv("JAX_METALLIB_LOOP_UNROLL");
        if (!v || v[0] == '\0') {
            return 8;
        }
        long k = std::strtol(v, nullptr, 10);
        return static_cast<int>(std::max(1L, std::min(k, 32L)));
    }();
    return value;
}

static int ChooseLoopUnroll(mlir::stablehlo::WhileOp whileOp, mlir::Block& bodyBlock,
                            const CountedWhile& counted,
                            const std::unordered_set<mlir::Operation*>* allowedWrites = nullptr) {
    const int maxUnroll = MaxLoopUnroll();
    if (maxUnroll <= 1 || !counted.upper_bound_is_const || counted.step <= 0) {
        return 1;
    }
    int64_t lower = 0;
    if (!ReadConstantIntScalar(whileOp->getOperand(counted.counter_index), &lower)) {
        return 1;
    }
    const int64_t span = counted.upper_bound_const - lower;
    if (span <= 0) {
        return 1;
    }
    const int64_t trips = (span + counted.step - 1) / counted.step;
    // In-loop buffer writes (dynamic_update_slice, scatter, and scan output
    // stacking, which lowers to dynamic_update_slice) are full-buffer
    // mask/select updates on MPS; chaining several per iteration measured
    // slower (1024-row write loop: 12.9 ms -> 27.3 ms at unroll 8), so such
    // bodies are never unrolled. func.call is skipped since it may hide them.
    size_t numOps = 0;
    for (mlir::Operation& op : bodyBlock) {
        if (allowedWrites && allowedWrites->count(&op)) {
            ++numOps;
            continue;
        }
        if (mlir::isa<mlir::stablehlo::WhileOp, mlir::stablehlo::CaseOp, mlir::stablehlo::IfOp,
                      mlir::stablehlo::DynamicUpdateSliceOp, mlir::stablehlo::ScatterOp, mlir::func::CallOp>(
                op)) {
            return 1;
        }
        ++numOps;
    }
    if (numOps > 64) {
        return 1;
    }
    for (int k = maxUnroll; k >= 2; --k) {
        if (trips % k == 0 && trips >= 2 * static_cast<int64_t>(k)) {
            return k;
        }
    }
    return 1;
}

// Scan output stacking lowers to one dynamic_update_slice of a full row per
// step into a carried buffer, and each such write is a full-buffer
// mask/select on MPS (256-step tanh RNN: 6.3 ms with stacked outputs vs
// 2.9 ms carry-only). Plain unrolling chained U full-buffer writes and was
// slower, so instead: when a buffer carry feeds only its write and the write
// result is only returned, evaluate the body U times against the unchanged
// buffer, collect the U rows and start indices, and apply them in one pass
// (later sub-iterations win on duplicate rows; starts clamp like XLA).
// JAX_METALLIB_DISABLE_BATCHED_LOOP_WRITES=1 keeps the per-step writes.
struct BatchedLoopWrite {
    unsigned carry_index = 0;
    mlir::Value update;
    mlir::Value start;
    int64_t rows = 0;
    // Axis the one-slice writes run along (0 for scan stacking; 1 for a
    // [1, T, H, D] KV-cache fill). Every other dim of the update spans the
    // buffer, so its start clamps to 0 and is ignored.
    int64_t axis = 0;
};

static bool CollectBatchableLoopWrites(mlir::Block& bodyBlock, const CountedWhile& counted,
                                       std::vector<BatchedLoopWrite>* out,
                                       std::unordered_set<mlir::Operation*>* ops) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_BATCHED_LOOP_WRITES")) {
        return false;
    }
    mlir::Operation* term = bodyBlock.getTerminator();
    if (!term) {
        return false;
    }
    for (mlir::Operation& op : bodyBlock) {
        auto dus = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(&op);
        if (!dus) {
            continue;
        }
        auto ba = mlir::dyn_cast<mlir::BlockArgument>(dus->getOperand(0));
        if (!ba || ba.getOwner() != &bodyBlock || !ba.hasOneUse()) {
            return false;
        }
        const unsigned k = ba.getArgNumber();
        if (k == counted.counter_index) {
            return false;
        }
        mlir::Value res = dus->getResult(0);
        if (!res.hasOneUse()) {
            return false;
        }
        mlir::OpOperand& use = *res.use_begin();
        if (use.getOwner() != term || use.getOperandNumber() != k) {
            return false;
        }
        auto bufT = mlir::dyn_cast<mlir::RankedTensorType>(dus->getOperand(0).getType());
        auto updT = mlir::dyn_cast<mlir::RankedTensorType>(dus->getOperand(1).getType());
        if (!bufT || !updT || !bufT.hasStaticShape() || !updT.hasStaticShape() || bufT.getRank() < 1 ||
            updT.getRank() != bufT.getRank() || !bufT.getElementType().isF32() ||
            updT.getElementType() != bufT.getElementType()) {
            return false;
        }
        int64_t axis = -1;
        for (int64_t d = 0; d < bufT.getRank(); ++d) {
            if (updT.getDimSize(d) == bufT.getDimSize(d)) {
                continue;
            }
            if (axis >= 0 || updT.getDimSize(d) != 1) {
                return false;
            }
            axis = d;
        }
        if (axis < 0 || bufT.getDimSize(axis) > (1 << 20)) {
            return false;
        }
        if (dus->getNumOperands() != 2 + static_cast<unsigned>(bufT.getRank())) {
            return false;
        }
        BatchedLoopWrite w;
        w.carry_index = k;
        w.update = dus->getOperand(1);
        w.start = dus->getOperand(2 + static_cast<unsigned>(axis));
        w.rows = bufT.getDimSize(axis);
        w.axis = axis;
        out->push_back(w);
        ops->insert(&op);
    }
    return !out->empty();
}

// Writes rows[u] (size 1 along axis) at clamped start starts[u] into buffer,
// in order, as one gather + select along axis: each slice takes the last
// sub-iteration that targets it, or keeps its value.
static MPSGraphTensor* ApplyBatchedRowWrites(MPSGraph* g, MPSGraphTensor* buffer,
                                             NSArray<MPSGraphTensor*>* rows, NSArray<MPSGraphTensor*>* starts,
                                             int64_t numRows, int64_t axis) {
    const NSUInteger U = rows.count;
    MPSGraphTensor* rowsCat = [g concatTensors:rows dimension:(NSInteger)axis name:nil];
    NSMutableArray<MPSGraphTensor*>* idx = [NSMutableArray arrayWithCapacity:U];
    for (MPSGraphTensor* s in starts) {
        MPSGraphTensor* t =
            s.dataType == MPSDataTypeInt32 ? s : [g castTensor:s toType:MPSDataTypeInt32 name:nil];
        [idx addObject:[g reshapeTensor:t withShape:@[@1, @1] name:nil]];
    }
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* one = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* last = [g constantWithScalar:(double)(numRows - 1) dataType:MPSDataTypeInt32];
    MPSGraphTensor* idxRow = [g concatTensors:idx dimension:1 name:nil];
    idxRow = [g maximumWithPrimaryTensor:idxRow secondaryTensor:zero name:nil];
    idxRow = [g minimumWithPrimaryTensor:idxRow secondaryTensor:last name:nil];

    std::vector<int32_t> iota(static_cast<size_t>(numRows));
    for (int64_t r = 0; r < numRows; ++r) {
        iota[static_cast<size_t>(r)] = static_cast<int32_t>(r);
    }
    std::vector<int32_t> order(U);
    for (NSUInteger u = 0; u < U; ++u) {
        order[u] = static_cast<int32_t>(u + 1);
    }
    MPSGraphTensor* iotaT = [g constantWithData:[NSData dataWithBytes:iota.data() length:iota.size() * 4]
                                          shape:@[@(numRows), @1]
                                       dataType:MPSDataTypeInt32];
    MPSGraphTensor* orderT = [g constantWithData:[NSData dataWithBytes:order.data() length:order.size() * 4]
                                           shape:@[@1, @(U)]
                                        dataType:MPSDataTypeInt32];
    MPSGraphTensor* eq = [g equalWithPrimaryTensor:iotaT secondaryTensor:idxRow name:nil];
    MPSGraphTensor* who = [g selectWithPredicateTensor:eq
                                   truePredicateTensor:orderT
                                  falsePredicateTensor:zero
                                                  name:nil];
    who = [g reductionMaximumWithTensor:who axis:1 name:nil];
    who = [g reshapeTensor:who withShape:@[@(numRows)] name:nil];
    MPSGraphTensor* hit = [g greaterThanWithPrimaryTensor:who secondaryTensor:zero name:nil];
    MPSGraphTensor* sel = [g subtractionWithPrimaryTensor:who secondaryTensor:one name:nil];
    sel = [g maximumWithPrimaryTensor:sel secondaryTensor:zero name:nil];
    MPSGraphTensor* gathered = [g gatherWithUpdatesTensor:rowsCat
                                            indicesTensor:sel
                                                     axis:(NSUInteger)axis
                                          batchDimensions:0
                                                     name:nil];
    NSMutableArray<NSNumber*>* hitShape = [NSMutableArray arrayWithCapacity:buffer.shape.count];
    for (NSUInteger d = 0; d < buffer.shape.count; ++d) {
        [hitShape addObject:(d == (NSUInteger)axis ? @(numRows) : @1)];
    }
    hit = [g reshapeTensor:hit withShape:hitShape name:nil];
    return [g selectWithPredicateTensor:hit
                    truePredicateTensor:gathered
                   falsePredicateTensor:buffer
                                   name:nil];
}

static NSArray<MPSGraphTensor*>*
EvaluateBatchedWriteBody(MPSGraph* graph, mlir::Block& bodyBlock, const ValueMap& values,
                         NSArray<MPSGraphTensor*>* iterationArguments, mlir::ModuleOp module, int depth,
                         int unroll, const std::vector<BatchedLoopWrite>& writes, std::string* blockError,
                         BlockProcessor processBlock) {
    std::vector<mlir::Value> capture;
    for (const auto& w : writes) {
        capture.push_back(w.update);
        capture.push_back(w.start);
    }
    std::vector<NSMutableArray<MPSGraphTensor*>*> rows(writes.size());
    std::vector<NSMutableArray<MPSGraphTensor*>*> starts(writes.size());
    for (size_t i = 0; i < writes.size(); ++i) {
        rows[i] = [NSMutableArray arrayWithCapacity:(NSUInteger)unroll];
        starts[i] = [NSMutableArray arrayWithCapacity:(NSUInteger)unroll];
    }
    NSMutableArray<MPSGraphTensor*>* args = [NSMutableArray arrayWithArray:iterationArguments];
    for (int u = 0; u < unroll; ++u) {
        std::vector<MPSGraphTensor*> captured;
        NSArray<MPSGraphTensor*>* next =
            EvaluateLoopBody(graph, bodyBlock, values, args, module, depth, true, blockError, processBlock,
                             nullptr, &capture, &captured);
        if (!blockError->empty()) {
            return iterationArguments;
        }
        if (captured.size() != capture.size() || next.count != iterationArguments.count) {
            *blockError = "while body: batched write capture mismatch";
            return iterationArguments;
        }
        args = [NSMutableArray arrayWithArray:next];
        for (size_t i = 0; i < writes.size(); ++i) {
            [rows[i] addObject:captured[2 * i]];
            [starts[i] addObject:captured[2 * i + 1]];
            args[writes[i].carry_index] = iterationArguments[writes[i].carry_index];
        }
    }
    for (size_t i = 0; i < writes.size(); ++i) {
        const unsigned k = writes[i].carry_index;
        args[k] = ApplyBatchedRowWrites(graph, iterationArguments[k], rows[i], starts[i], writes[i].rows,
                                        writes[i].axis);
    }
    return args;
}

// True when v is a constant, possibly behind broadcast_in_dim / reshape /
// convert (JAX emits `x + 0.0` on a tensor as add(x, broadcast(constant))).
static bool IsConstantLikeValue(mlir::Value v) {
    for (int hops = 0; hops < 4; ++hops) {
        mlir::Operation* od = v.getDefiningOp();
        if (!od) {
            return false;
        }
        if (od->hasTrait<mlir::OpTrait::ConstantLike>()) {
            return true;
        }
        if (!mlir::isa<mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::ReshapeOp,
                       mlir::stablehlo::ConvertOp>(od) ||
            od->getNumOperands() != 1) {
            return false;
        }
        v = od->getOperand(0);
    }
    return false;
}

// Follows v back through single-result ops whose other operands are all
// constant-like (x + 0, x * 1, copies), which MPSGraph may fold to x.
static mlir::Value TraceForwardedValue(mlir::Value v) {
    for (int hops = 0; hops < 8; ++hops) {
        mlir::Operation* def = v.getDefiningOp();
        if (!def || def->getNumResults() != 1) {
            return v;
        }
        mlir::Value only;
        for (mlir::Value operand : def->getOperands()) {
            if (IsConstantLikeValue(operand)) {
                continue;
            }
            if (only) {
                return v;
            }
            only = operand;
        }
        if (!only || only.getType() != v.getType()) {
            return v;
        }
        v = only;
    }
    return v;
}

// MPSGraph forLoopWithLowerBound updates carries as if in slot order: a body
// output that is (or folds to) another slot's iteration argument read that
// slot's new value when the source slot came first. fori_loop (a, b) ->
// (2a + 1, a) returned b = new a; (a, b) -> (b, 2b + 1) was right. Such
// outputs get their own tensor. Checked on tensors (also catches permutations
// composed over unrolled sub-iterations) and, when the body is evaluated once
// per iteration, on the MLIR return values (catches a + 0.0 folded back to a).
// JAX_METALLIB_DISABLE_LOOP_CARRY_COPY=1 disables this.
static NSArray<MPSGraphTensor*>* DecoupleForwardedCarries(MPSGraph* g, mlir::Block& body,
                                                          NSArray<MPSGraphTensor*>* iterArgs,
                                                          NSArray<MPSGraphTensor*>* outs, bool traceBody) {
    if (!outs || outs.count != iterArgs.count || EnvFlagEnabled("JAX_METALLIB_DISABLE_LOOP_CARRY_COPY")) {
        return outs;
    }
    auto ret = mlir::dyn_cast_or_null<mlir::stablehlo::ReturnOp>(body.getTerminator());
    NSMutableArray<MPSGraphTensor*>* fixed = nil;
    for (NSUInteger i = 0; i < outs.count; ++i) {
        bool alias = false;
        for (NSUInteger j = 0; j < iterArgs.count && !alias; ++j) {
            alias = j != i && outs[i] == iterArgs[j] && outs[j] != iterArgs[j];
        }
        if (!alias && traceBody && ret && i < ret->getNumOperands()) {
            auto ba = mlir::dyn_cast<mlir::BlockArgument>(TraceForwardedValue(ret->getOperand((unsigned)i)));
            if (ba && ba.getOwner() == &body && ba.getArgNumber() != i) {
                const unsigned j = ba.getArgNumber();
                alias = j < outs.count && outs[j] != iterArgs[j];
            }
        }
        if (!alias) {
            continue;
        }
        if (!fixed) {
            fixed = [NSMutableArray arrayWithArray:outs];
        }
        fixed[i] = [g identityWithTensor:outs[i] name:nil];
    }
    return fixed ? fixed : outs;
}

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
        // Promoting dynamic_update_slice carries to MPSGraph variables silently
        // dropped every in-loop update: a scan over a >=64 KiB buffer returned
        // its initial value (tests/test_scan_update.py). Root cause is not yet
        // isolated, so keep it opt-in until the variable path is proven correct.
        const bool enable_variable_carry = EnvFlagEnabled("JAX_METALLIB_ENABLE_VARIABLE_CARRY") &&
                                           !EnvFlagEnabled("JAX_METALLIB_DISABLE_VARIABLE_CARRY");

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
        int unroll = variables ? 1 : ChooseLoopUnroll(whileOp, bodyBlock, counted);
        std::vector<BatchedLoopWrite> batchWrites;
        if (unroll == 1 && !variables) {
            std::unordered_set<mlir::Operation*> writeOps;
            if (CollectBatchableLoopWrites(bodyBlock, counted, &batchWrites, &writeOps)) {
                unroll = ChooseLoopUnroll(whileOp, bodyBlock, counted, &writeOps);
            }
            if (unroll == 1) {
                batchWrites.clear();
            }
        }
        MPSGraphTensor* step = [graph constantWithScalar:counted.step * unroll dataType:lowerBound.dataType];

        NSArray<MPSGraphTensor*>* outputs = [graph
            forLoopWithLowerBound:lowerBound
                       upperBound:upperBound
                             step:step
             initialBodyArguments:initialInputs
                             body:^NSArray<MPSGraphTensor*>*(MPSGraphTensor*,
                                                             NSArray<MPSGraphTensor*>* iterationArguments) {
                               if (!batchWrites.empty()) {
                                   NSArray<MPSGraphTensor*>* batched = EvaluateBatchedWriteBody(
                                       graph, bodyBlock, values, iterationArguments, module, depth, unroll,
                                       batchWrites, &blockError, processBlock);
                                   return DecoupleForwardedCarries(graph, bodyBlock, iterationArguments,
                                                                   batched, false);
                               }
                               NSArray<MPSGraphTensor*>* args = iterationArguments;
                               for (int u = 0; u < unroll && blockError.empty(); ++u) {
                                   args = EvaluateLoopBody(graph, bodyBlock, values, args, module, depth,
                                                           true, &blockError, processBlock, variables);
                               }
                               return DecoupleForwardedCarries(graph, bodyBlock, iterationArguments, args,
                                                               unroll == 1);
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
    outputs = TieWhileOutputs(graph, outputs);

    for (NSUInteger i = 0; i < outputs.count; ++i) {
        ctx.values[whileOp->getResult(i).getAsOpaquePointer()] = outputs[i];
    }
    return ProcessResult{};
}

// Whether a case's branches do enough work that running only the taken one
// (MPSGraph if) beats building both and selecting. The if materialises its
// results, which cost more than cheap branches: A/B, one cond on f32[4M]
// 1.12-1.15 ms as if vs 0.83-0.84 ms as select, 100 conds in a fori_loop
// 1.39-1.41 vs 1.11-1.12 ms; an untaken matmul branch 0.48-0.62 vs 0.75-0.96.
static bool CaseBranchesAreHeavy(mlir::Operation* caseOp) {
    constexpr size_t kHeavyOpCount = 64;
    bool heavy = false;
    size_t count = 0;
    for (mlir::Region& region : caseOp->getRegions()) {
        region.walk([&](mlir::Operation* op) {
            ++count;
            llvm::StringRef name = op->getName().getStringRef();
            if (name == "stablehlo.while" || name == "stablehlo.case" || name == "stablehlo.dot_general" ||
                name == "stablehlo.dot" || name == "stablehlo.convolution" || name == "stablehlo.sort" ||
                name == "stablehlo.fft" || name == "stablehlo.custom_call" || name == "func.call") {
                heavy = true;
            }
        });
    }
    return heavy || count > kHeavyOpCount;
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

    bool branchInRegion = false;
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
        branchCtx.in_control_flow = ctx.in_control_flow || branchInRegion;
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

    // lax.cond lowers to a two-branch case (index 0 takes branch 0; StableHLO
    // sends any other index to the last branch). Building both branches and
    // selecting (below) runs both on every execution, so a cond-guarded body in
    // a loop never skips work: the n >= 64 Schur sweep paid a full QR sweep on
    // every capped iteration and hit the completion timeout. MPSGraph's if runs
    // only the taken branch, but it costs more than select for cheap branches,
    // so it is used only when a branch is heavy (CaseBranchesAreHeavy).
    // JAX_METALLIB_DISABLE_CASE_GRAPH_IF=1 restores evaluate-both + select for
    // every case.
    static const bool disableGraphIf = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_CASE_GRAPH_IF");
        return v != nullptr && v[0] == '1';
    }();
    if (numBranches == 2 && numResults > 0 && !disableGraphIf && CaseBranchesAreHeavy(ctx.op)) {
        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:selector.dataType];
        MPSGraphTensor* pred = [ctx.graph notEqualWithPrimaryTensor:selector secondaryTensor:zero name:nil];
        if (pred.shape.count > 0) {
            pred = [ctx.graph reshapeTensor:pred withShape:@[] name:nil];
        }
        // A failed branch must still yield tensors of the result types.
        auto placeholderOutputs = [&]() -> NSArray<MPSGraphTensor*>* {
            NSMutableArray<MPSGraphTensor*>* out = [NSMutableArray arrayWithCapacity:(NSUInteger)numResults];
            for (size_t r = 0; r < numResults; ++r) {
                auto type = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getResult((unsigned)r).getType());
                MPSDataType dt = type ? MlirTypeToMps(type.getElementType()) : MPSDataTypeFloat32;
                [out addObject:[ctx.graph constantWithScalar:0
                                                       shape:GetOutputShape(ctx.op, (unsigned)r)
                                                    dataType:dt]];
            }
            return out;
        };
        std::string ifError;
        std::string* ifErrorPtr = &ifError;
        auto buildBranch = [&](size_t b) -> NSArray<MPSGraphTensor*>* {
            if (!ifErrorPtr->empty()) {
                return placeholderOutputs();
            }
            branchInRegion = true;
            NSArray<MPSGraphTensor*>* out = nil;
            {
                ScopedControlFlowContext cf_guard;
                out = evaluateBranch(b, ifErrorPtr);
            }
            branchInRegion = false;
            if (!out) {
                if (ifErrorPtr->empty()) {
                    *ifErrorPtr = "stablehlo.case branch evaluation failed";
                }
                return placeholderOutputs();
            }
            return out;
        };
        NSArray<MPSGraphTensor*>* outputs = [ctx.graph ifWithPredicateTensor:pred
            thenBlock:^NSArray<MPSGraphTensor*>*() {
              return buildBranch(1);
            }
            elseBlock:^NSArray<MPSGraphTensor*>*() {
              return buildBranch(0);
            }
            name:nil];
        if (!ifError.empty()) {
            return ProcessResult::Error(ifError);
        }
        if (!outputs || outputs.count != (NSUInteger)numResults) {
            return ProcessResult::Error("stablehlo.case: ifWithPredicateTensor returned wrong arity");
        }
        for (size_t r = 0; r < numResults; ++r) {
            ctx.values[ctx.op->getResult((unsigned)r).getAsOpaquePointer()] = outputs[r];
        }
        return ProcessResult{};
    }

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

// stablehlo.optimization_barrier had no handler, so every jax.checkpoint
// (remat) gradient under jit failed. It only constrains XLA's scheduling;
// semantically it is the identity over its (variadic) operands.
static ProcessResult HandleOptimizationBarrier(HandlerContext& ctx) {
    const unsigned numOperands = ctx.op->getNumOperands();
    if (numOperands != ctx.op->getNumResults()) {
        return ProcessResult::Error("optimization_barrier: operand/result arity mismatch");
    }
    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* t = GetInputTensor(ctx, i);
        if (!t) {
            return ProcessResult::Error("optimization_barrier: missing operand tensor " + std::to_string(i));
        }
        SetOutputTensor(ctx.values, ctx.op, t, i);
    }
    return ProcessResult{};
}

REGISTER_MPS_OP("stablehlo.while", HandleWhileOp);
REGISTER_MPS_OP("stablehlo.case", HandleCaseOp);
REGISTER_MPS_OP("stablehlo.optimization_barrier", HandleOptimizationBarrier);

REGISTER_MPS_OP("stablehlo.custom_call", HandleCustomCall);

} // namespace jax_metallib
