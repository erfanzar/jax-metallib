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

#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace jax_metallib {

namespace {

// Names the reducer only when the body is exactly `return op(arg0, arg1)` (either
// operand order). This used to return the first op of ANY body, so a multi-op
// combiner was lowered as that op: lax.cumlogsumexp's logaddexp region starts
// with stablehlo.maximum and came back as a running max (maxabs 1.1 vs CPU).
// Other bodies return "unsupported_region(<ops>)", which every caller rejects.
// A body with no ops (scatter `set`) still returns "".
std::string GetReductionOpType(mlir::Region& body) {
    if (body.empty())
        return "";

    mlir::Block& block = body.front();
    mlir::Operation* reducer = nullptr;
    std::string ops;
    size_t count = 0;
    for (mlir::Operation& op : block) {
        std::string opName = op.getName().getStringRef().str();
        if (opName == "stablehlo.return")
            continue;
        ops += (count++ ? "," : "") + opName;
        reducer = &op;
    }
    if (count == 0)
        return "";

    mlir::Operation* terminator = block.empty() ? nullptr : &block.back();
    bool simple = count == 1 && block.getNumArguments() == 2 && reducer->getNumOperands() == 2 &&
                  reducer->getNumResults() == 1 && terminator && terminator->getNumOperands() == 1 &&
                  terminator->getOperand(0) == reducer->getResult(0);
    if (simple) {
        mlir::Value a = reducer->getOperand(0);
        mlir::Value b = reducer->getOperand(1);
        mlir::Value x = block.getArgument(0);
        mlir::Value y = block.getArgument(1);
        simple = (a == x && b == y) || (a == y && b == x);
    }
    if (!simple)
        return "unsupported_region(" + ops + ")";
    return reducer->getName().getStringRef().str();
}

enum class ArgReduceKind { kUnknown, kMax, kMin };

bool IsBlockArg(mlir::Value value, mlir::Block& block, unsigned index) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == index;
}

ArgReduceKind detectArgReduceKind(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return ArgReduceKind::kUnknown;
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() < 4) {
        return ArgReduceKind::kUnknown;
    }

    for (mlir::Operation& nestedOp : body) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }

        bool forwardValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 0) && IsBlockArg(compareOp.getRhs(), body, 2);
        bool reversedValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 2) && IsBlockArg(compareOp.getRhs(), body, 0);
        if (!forwardValueCompare && !reversedValueCompare) {
            continue;
        }

        auto dir = compareOp.getComparisonDirection();
        bool lhsWins = false;
        if (dir == mlir::stablehlo::ComparisonDirection::GT ||
            dir == mlir::stablehlo::ComparisonDirection::GE) {
            lhsWins = true;
        } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                   dir == mlir::stablehlo::ComparisonDirection::LE) {
            lhsWins = false;
        } else {
            continue;
        }

        if (reversedValueCompare) {
            lhsWins = !lhsWins;
        }
        return lhsWins ? ArgReduceKind::kMax : ArgReduceKind::kMin;
    }
    return ArgReduceKind::kUnknown;
}

std::string JoinInts(llvm::ArrayRef<int64_t> values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::string JoinPadding(const std::vector<std::pair<int64_t, int64_t>>& padding) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < padding.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << "[" << padding[i].first << ", " << padding[i].second << "]";
    }
    oss << "]";
    return oss.str();
}

bool ReadArrayOrDefaultOnes(std::optional<llvm::ArrayRef<int64_t>> attr, size_t rank,
                            std::vector<int64_t>& out) {
    out.assign(rank, 1);
    if (!attr.has_value()) {
        return true;
    }
    if (attr->size() != rank) {
        return false;
    }
    out.assign(attr->begin(), attr->end());
    return true;
}

bool ReadPaddingOrDefaultZeros(std::optional<mlir::DenseIntElementsAttr> paddingAttr, size_t rank,
                               std::vector<std::pair<int64_t, int64_t>>& out) {
    out.clear();
    out.resize(rank, {0, 0});
    if (!paddingAttr.has_value()) {
        return true;
    }

    std::vector<int64_t> flatPadding;
    flatPadding.reserve(rank * 2);
    for (int64_t value : paddingAttr->getValues<int64_t>()) {
        flatPadding.push_back(value);
    }
    if (flatPadding.size() != rank * 2) {
        return false;
    }

    for (size_t i = 0; i < rank; ++i) {
        out[i] = {flatPadding[2 * i], flatPadding[2 * i + 1]};
    }
    return true;
}

std::string DescribeReduceWindowPattern(mlir::stablehlo::ReduceWindowOp op,
                                        llvm::ArrayRef<int64_t> windowDims,
                                        llvm::ArrayRef<int64_t> windowStrides,
                                        llvm::ArrayRef<int64_t> baseDilations,
                                        llvm::ArrayRef<int64_t> windowDilations,
                                        const std::vector<std::pair<int64_t, int64_t>>& padding,
                                        const std::string& reducerType) {
    std::ostringstream oss;
    oss << "reducer=" << reducerType << ", window_dimensions=" << JoinInts(windowDims)
        << ", window_strides=" << JoinInts(windowStrides) << ", base_dilations=" << JoinInts(baseDilations)
        << ", window_dilations=" << JoinInts(windowDilations) << ", padding=" << JoinPadding(padding)
        << ", input_shape=";
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs().front().getType());
    if (!inputType) {
        oss << "<unranked>";
    } else {
        oss << JoinInts(inputType.getShape());
    }
    return oss.str();
}

bool GetScalarConstantAttr(mlir::Value value, mlir::DenseElementsAttr& outDense) {
    outDense = {};
    if (!value) {
        return false;
    }
    auto* defOp = value.getDefiningOp();
    // JAX often wraps a scalar init in a rank-0 broadcast_in_dim.
    if (auto broadcast = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(defOp)) {
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
        if (!resultType || resultType.getRank() != 0) {
            return false;
        }
        defOp = broadcast->getOperand(0).getDefiningOp();
    }
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
    if (!constOp) {
        return false;
    }
    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!denseAttr) {
        return false;
    }
    if (denseAttr.getNumElements() != 1) {
        return false;
    }
    outDense = denseAttr;
    return true;
}

bool IsScalarZero(mlir::Value value) {
    mlir::DenseElementsAttr dense;
    if (!GetScalarConstantAttr(value, dense)) {
        return false;
    }
    auto elemType = dense.getElementType();
    if (mlir::isa<mlir::FloatType>(elemType)) {
        llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
        return ap.isZero();
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        llvm::APInt ap = *dense.getValues<llvm::APInt>().begin();
        return ap.isZero();
    }
    return false;
}

// Scalar float constant equal to +0 (not -0).
bool IsScalarPositiveFloatZero(mlir::Value value) {
    mlir::DenseElementsAttr dense;
    if (!GetScalarConstantAttr(value, dense) || !mlir::isa<mlir::FloatType>(dense.getElementType())) {
        return false;
    }
    llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
    return ap.isZero() && !ap.isNegative();
}

// Scalar float constant that is not +-0 (e.g. the -inf/+inf init of a max/min
// window), so it can never tie with a zero.
bool IsNonZeroFloatScalar(mlir::Value value) {
    mlir::DenseElementsAttr dense;
    if (!GetScalarConstantAttr(value, dense) || !mlir::isa<mlir::FloatType>(dense.getElementType())) {
        return false;
    }
    return !(*dense.getValues<llvm::APFloat>().begin()).isZero();
}

// Scalar init that leaves every addend unchanged: integer 0 or float -0. A +0
// float init is not an identity (+0 + -0 == +0), so a window summing to -0
// must still add it (lax.reduce_window(x, 0., add) of [-0, -0] is +0).
bool IsExactAddIdentityInit(mlir::Value value) {
    return IsScalarZero(value) && !IsScalarPositiveFloatZero(value);
}

} // namespace

constexpr int64_t kTreeReduceThreshold = 1024;
constexpr int64_t kTreeReduceWindowSize = 256;

int64_t GetTotalElements(NSArray<NSNumber*>* shape) {
    if (!shape) return 0;
    int64_t total = 1;
    for (NSNumber* dim in shape) {
        total *= [dim longLongValue];
    }
    return total;
}

static bool IsFloatForNaN(MPSDataType t) {
    return t == MPSDataTypeFloat32 || t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
}

// XLA orders -0 < +0 in max/min. MPSGraph max/min pooling, cumulative max/min
// and plain maximum/minimum keep one operand on a +-0 tie, so reduce_window
// max/min (lax.cummax/cummin, sliding/strided/pooled windows) returned the
// losing zero (build/probe_r20c_ties.py: every case failed).
// JAX_METALLIB_DISABLE_WINDOW_ZERO_TIE=1 restores the plain lowerings.
static bool WindowZeroTieEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_WINDOW_ZERO_TIE");
        return !(v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'));
    }();
    return enabled;
}

// XLA:CPU float window max/min flushes f32 subnormals to a signed zero (see
// FlushF32SubnormalSigned). JAX_METALLIB_DISABLE_WINDOW_SUBNORMAL_FLUSH=1 skips it.
static bool WindowSubnormalFlushEnabled() {
    static const bool disabled = []() {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_WINDOW_SUBNORMAL_FLUSH");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return !disabled;
}

// Scalar f32 constant that is a subnormal; *negative receives its sign.
static bool IsF32SubnormalScalar(mlir::Value value, bool* negative) {
    mlir::DenseElementsAttr dense;
    if (!value || !GetScalarConstantAttr(value, dense) || !dense.getElementType().isF32()) {
        return false;
    }
    llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
    if (!ap.isDenormal()) {
        return false;
    }
    *negative = ap.isNegative();
    return true;
}

// Scalar f32 constant -inf (negative) or +inf.
static bool IsF32InfScalar(mlir::Value value, bool negative) {
    mlir::DenseElementsAttr dense;
    if (!value || !GetScalarConstantAttr(value, dense) || !dense.getElementType().isF32()) {
        return false;
    }
    llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
    return ap.isInfinity() && ap.isNegative() == negative;
}

enum class InitZeroKind { kUnknown, kNonZero, kPosZero, kNegZero };

// Kind of a scalar float constant init (kUnknown when not a constant). A
// subnormal f32 init is the signed zero XLA:CPU flushes it to; the f32 window
// entry feeds that zero in its place (HandleReduceWindowFloat). A reduce can
// normalize its init independently of the window-only flush switch.
static InitZeroKind GetInitZeroKind(mlir::Value v, bool flushF32Subnormal = false) {
    if (!v) {
        return InitZeroKind::kUnknown;
    }
    bool subNeg = false;
    if ((flushF32Subnormal || WindowSubnormalFlushEnabled()) && IsF32SubnormalScalar(v, &subNeg)) {
        return subNeg ? InitZeroKind::kNegZero : InitZeroKind::kPosZero;
    }
    if (IsNonZeroFloatScalar(v)) {
        return InitZeroKind::kNonZero;
    }
    if (IsScalarPositiveFloatZero(v)) {
        return InitZeroKind::kPosZero;
    }
    return IsScalarZero(v) ? InitZeroKind::kNegZero : InitZeroKind::kUnknown;
}

// NaN-propagating max/min of two tensors with XLA's -0 < +0 tie rule. Where
// both are zero, max is a + b (-0 only if both are -0) and min is
// -((-a) + (-b)) (-0 if either is -0). `bv` is the MLIR value behind b when
// it is the op's init: a non-zero constant needs no fix-up, and a +-0 constant
// must not use the sum rule, because MPSGraph folds x + (+-0 constant) to x
// (build/probe_r20c_padlow.py: max(x, +0 init) gave -0 for x == -0).
static MPSGraphTensor* WindowMaxMin(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool isMax,
                                    mlir::Value bv = mlir::Value(), bool flushF32Subnormal = false) {
    if (!IsFloatForNaN(a.dataType)) {
        return isMax ? [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil]
                     : [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* m = isMax ? [g maximumWithNaNPropagationWithPrimaryTensor:a secondaryTensor:b name:nil]
                              : [g minimumWithNaNPropagationWithPrimaryTensor:a secondaryTensor:b name:nil];
    const InitZeroKind kind = GetInitZeroKind(bv, flushF32Subnormal);
    if (kind == InitZeroKind::kNonZero || !WindowZeroTieEnabled() || b.dataType != a.dataType) {
        return m;
    }
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:a.dataType];
    if (kind == InitZeroKind::kPosZero || kind == InitZeroKind::kNegZero) {
        // Plain max/min return the constant on a tie, which is right when it
        // is the winning zero; otherwise keep a wherever a is a zero.
        if (kind == (isMax ? InitZeroKind::kPosZero : InitZeroKind::kNegZero)) {
            return m;
        }
        MPSGraphTensor* aZero = [g equalWithPrimaryTensor:a secondaryTensor:zero name:nil];
        return [g selectWithPredicateTensor:aZero truePredicateTensor:a falsePredicateTensor:m name:nil];
    }
    MPSGraphTensor* both =
        [g logicalANDWithPrimaryTensor:[g equalWithPrimaryTensor:a secondaryTensor:zero name:nil]
                       secondaryTensor:[g equalWithPrimaryTensor:b secondaryTensor:zero name:nil]
                                  name:nil];
    MPSGraphTensor* fixed =
        isMax ? [g additionWithPrimaryTensor:a secondaryTensor:b name:nil]
              : [g negativeWithTensor:[g additionWithPrimaryTensor:[g negativeWithTensor:a name:nil]
                                                   secondaryTensor:[g negativeWithTensor:b name:nil]
                                                              name:nil]
                                 name:nil];
    return [g selectWithPredicateTensor:both truePredicateTensor:fixed falsePredicateTensor:m name:nil];
}

// JAX_METALLIB_LEGACY_ARGREDUCE_NAN=1 restores the int32-mask NaN pass for
// float argmax/argmin (two extra full reductions; f32 [4096,4096] ~3.3 ms).
static bool LegacyArgReduceNaN() {
    static const bool legacy = [] {
        const char* v = std::getenv("JAX_METALLIB_LEGACY_ARGREDUCE_NAN");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return legacy;
}

MPSGraphTensor* CreateReduction(MPSGraph* graph, MPSGraphTensor* input, NSArray<NSNumber*>* axes,
                                const std::string& reductionType) {
    if (reductionType == "stablehlo.add") {
        return [graph reductionSumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        return [graph reductionProductWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        // StableHLO max/min reductions propagate NaN; the plain MPSGraph
        // reductions returned the largest finite value instead.
        if (IsFloatForNaN(input.dataType)) {
            return [graph reductionMaximumPropagateNaNWithTensor:input axes:axes name:nil];
        }
        return [graph reductionMaximumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        if (IsFloatForNaN(input.dataType)) {
            return [graph reductionMinimumPropagateNaNWithTensor:input axes:axes name:nil];
        }
        return [graph reductionMinimumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.and") {
        return [graph reductionAndWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.or") {
        return [graph reductionOrWithTensor:input axes:axes name:nil];
    }
    return nil;
}

MPSGraphTensor* TreeReduce(HandlerContext& ctx, MPSGraphTensor* input,
                           NSArray<NSNumber*>* reduceAxes,
                           const std::string& reductionType) {
    int64_t totalElements = GetTotalElements(input.shape);
    if (totalElements <= kTreeReduceThreshold) {
        return nil;
    }

    NSArray<NSNumber*>* originalShape = input.shape;
    NSUInteger rank = originalShape.count;
    int64_t numReduceDims = (int64_t)reduceAxes.count;

    if (numReduceDims == 0) {
        return nil;
    }

    bool fullReduce = (numReduceDims == (int64_t)rank);

    // The chunked form below builds one slice + reduce per 256-element window
    // and chains the chunk results serially: ~12K graph ops at 1M elements
    // (int32 sum 59 ms vs 1.2 ms for the native f32 path). Reduce through a
    // reshape tree instead: [k, W] -> reduce axis 1 -> [k] (+ tail), repeated.
    // JAX_METALLIB_LEGACY_TREE_REDUCE=1 restores the chunked form.
    static const bool legacyTreeReduce = [] {
        const char* v = std::getenv("JAX_METALLIB_LEGACY_TREE_REDUCE");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    if (fullReduce && !legacyTreeReduce) {
        const int64_t W = kTreeReduceWindowSize;
        MPSGraphTensor* cur = [ctx.graph reshapeTensor:input withShape:@[@(totalElements)] name:nil];
        int64_t len = totalElements;
        while (len > W) {
            const int64_t k = len / W;
            const int64_t mainLen = k * W;
            MPSGraphTensor* main = mainLen == len ? cur
                                                  : [ctx.graph sliceTensor:cur
                                                                    starts:@[@0]
                                                                      ends:@[@(mainLen)]
                                                                   strides:@[@1]
                                                                      name:nil];
            main = [ctx.graph reshapeTensor:main withShape:@[@(k), @(W)] name:nil];
            MPSGraphTensor* red = CreateReduction(ctx.graph, main, @[@1], reductionType);
            if (!red) {
                return nil;
            }
            red = [ctx.graph reshapeTensor:red withShape:@[@(k)] name:nil];
            int64_t next = k;
            if (mainLen < len) {
                MPSGraphTensor* tail = [ctx.graph sliceTensor:cur
                                                       starts:@[@(mainLen)]
                                                         ends:@[@(len)]
                                                      strides:@[@1]
                                                         name:nil];
                MPSGraphTensor* tailRed = CreateReduction(ctx.graph, tail, @[@0], reductionType);
                if (!tailRed) {
                    return nil;
                }
                tailRed = [ctx.graph reshapeTensor:tailRed withShape:@[@1] name:nil];
                // MPSGraph returned a rank-1 bool and/or reduction as si32 while
                // the reshaped main part stayed i1; concat rejects mixed types.
                if (tailRed.dataType != red.dataType) {
                    tailRed = [ctx.graph castTensor:tailRed toType:red.dataType name:nil];
                }
                red = [ctx.graph concatTensors:@[red, tailRed] dimension:0 name:nil];
                next += 1;
            }
            cur = red;
            len = next;
        }
        MPSGraphTensor* scalar = CreateReduction(ctx.graph, cur, @[@0], reductionType);
        if (scalar && scalar.dataType != input.dataType) {
            scalar = [ctx.graph castTensor:scalar toType:input.dataType name:nil];
        }
        return scalar ? [ctx.graph reshapeTensor:scalar withShape:@[] name:nil] : nil;
    }

    if (fullReduce) {
        int64_t totalFlat = totalElements;
        int64_t numWindowElements = kTreeReduceWindowSize;
        int64_t numChunks = (totalFlat + numWindowElements - 1) / numWindowElements;

        NSMutableArray<NSNumber*>* flatShape = [NSMutableArray arrayWithObject:@(totalFlat)];
        MPSGraphTensor* flat = [ctx.graph reshapeTensor:input withShape:flatShape name:nil];

        NSMutableArray<MPSGraphTensor*>* levelResults = [NSMutableArray array];

        for (int64_t i = 0; i < numChunks; i++) {
            int64_t start = i * numWindowElements;
            int64_t end = start + numWindowElements;
            if (end > totalFlat) end = totalFlat;

            MPSGraphTensor* startTensor = [ctx.graph constantWithScalar:start dataType:MPSDataTypeInt64];
            startTensor = [ctx.graph reshapeTensor:startTensor withShape:@[ @1 ] name:nil];
            MPSGraphTensor* endTensor = [ctx.graph constantWithScalar:end dataType:MPSDataTypeInt64];
            endTensor = [ctx.graph reshapeTensor:endTensor withShape:@[ @1 ] name:nil];

            // The modern MPSGraph slice API is sliceTensor:tensor starts:ends:strides:name:
            // with NSArray<NSNumber*> args (the older 3-arg sliceTensor:start:end: selector
            // no longer exists on macOS 15+). The input `flat` is rank-1 here.
            MPSGraphTensor* sliced = [ctx.graph sliceTensor:flat
                                                     starts:@[ @(start) ]
                                                       ends:@[ @(end) ]
                                                    strides:@[ @1 ]
                                                       name:nil];

            MPSGraphTensor* reduced = CreateReduction(ctx.graph, sliced, @[ @0 ], reductionType);
            if (!reduced) return nil;

            [levelResults addObject:reduced];
        }

        MPSGraphTensor* current = nil;
        for (MPSGraphTensor* chunkResult in levelResults) {
            if (!current) {
                current = chunkResult;
            } else {
                // Modern API: concatTensors:dimension:name: (the second keyword is 'dimension', not 'axis').
                current = CreateReduction(ctx.graph,
                                          [ctx.graph concatTensors:@[ current, chunkResult ] dimension:0 name:nil],
                                          @[ @0 ], reductionType);
            }
        }

        if (!current) return nil;

        while (current.shape.count > 0 && [current.shape[0] longLongValue] > 1) {
            int64_t currentSize = [current.shape[0] longLongValue];
            int64_t nextSize = (currentSize + 1) / 2;

            // Modern API: sliceTensor:tensor starts:ends:strides:name: with NSArray args.
            MPSGraphTensor* left = [ctx.graph sliceTensor:current
                                                   starts:@[ @0 ]
                                                     ends:@[ @(nextSize) ]
                                                  strides:@[ @1 ]
                                                     name:nil];
            MPSGraphTensor* right = [ctx.graph sliceTensor:current
                                                    starts:@[ @(nextSize) ]
                                                      ends:@[ @(currentSize) ]
                                                   strides:@[ @1 ]
                                                      name:nil];

            if ([right.shape[0] longLongValue] < nextSize) {
                int64_t padding = nextSize - [right.shape[0] longLongValue];
                MPSGraphTensor* paddingTensor = [ctx.graph constantWithScalar:0 dataType:current.dataType];
                paddingTensor = [ctx.graph tileTensor:paddingTensor withValues:@[@(padding)] name:nil];
                // Modern API: concatTensors:dimension:name:
                right = [ctx.graph concatTensors:@[ right, paddingTensor ] dimension:0 name:nil];
            }

            // Modern API: concatTensors:dimension:name:
            current = CreateReduction(ctx.graph,
                                      [ctx.graph concatTensors:@[ left, right ] dimension:0 name:nil],
                                      @[ @0 ], reductionType);
        }

        if ([current.shape[0] longLongValue] == 1) {
            current = [ctx.graph reshapeTensor:current withShape:@[] name:nil];
        }

        return current;
    }

    return nil;
}

// XLA:CPU float max/min reductions order -0 < +0 and flush f32 subnormals to a
// signed zero; MPSGraph's reductions return whichever zero they meet (f32
// [64,2048] axis 0: ~1000 bitwise diffs) and keep f32 subnormals. Reducing an
// order-preserving integer key matches XLA:CPU and costs ~1.0x on axis reduces
// (f16/bf16 use an int16 key; widening to f32 first cost 3.3x). Every select
// stays in the integer domain because MPSGraph's bf16 select flushes
// subnormals. NaN is detected from the bits, maps to INT_MAX (max) / INT_MIN
// (min) and comes back as the canonical qNaN, as the plain reduction did.
// f16/bf16 subnormals are kept (XLA:CPU keeps f16 always and bf16 from 4096
// elements). JAX_METALLIB_DISABLE_REDUCE_MAXMIN_KEY=1 restores the plain path.
static bool ReduceMaxMinKeyDisabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_REDUCE_MAXMIN_KEY");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return disabled;
}

// Scalar constant of the integer key type (int32 for f32, int16 for f16/bf16).
static MPSGraphTensor* OrderKeyConst(MPSGraph* g, MPSDataType it, int64_t v) {
    if (it == MPSDataTypeInt32) {
        const int32_t x = static_cast<int32_t>(v);
        return [g constantWithData:[NSData dataWithBytes:&x length:sizeof(x)] shape:@[@1] dataType:it];
    }
    const int16_t x = static_cast<int16_t>(v);
    return [g constantWithData:[NSData dataWithBytes:&x length:sizeof(x)] shape:@[@1] dataType:it];
}

// Order-preserving integer key of a float tensor: x >= 0 keeps its bits,
// x < 0 flips the magnitude bits, so -0 sorts just below +0 and integer
// max/min order every value like XLA. NaN maps to INT_MAX (max) / INT_MIN
// (min). With flushF32Subnormal an f32 subnormal becomes a signed zero first.
static MPSGraphTensor* FloatToOrderKey(MPSGraph* g, MPSGraphTensor* input, bool isMax,
                                       bool flushF32Subnormal) {
    const MPSDataType ft = input.dataType;
    const bool wide = ft == MPSDataTypeFloat32;
    const MPSDataType it = wide ? MPSDataTypeInt32 : MPSDataTypeInt16;
    auto k = [&](int64_t v) { return OrderKeyConst(g, it, v); };
    const int64_t mask = wide ? 0x7FFFFFFF : 0x7FFF;
    const int64_t signBit = wide ? INT32_MIN : INT16_MIN;
    const int64_t expMask = wide ? 0x7F800000 : (ft == MPSDataTypeFloat16 ? 0x7C00 : 0x7F80);
    const int64_t ext = isMax ? (wide ? INT32_MAX : INT16_MAX) : signBit;

    MPSGraphTensor* bits = [g reinterpretCastTensor:input toType:it name:nil];
    if (wide && flushF32Subnormal) {
        MPSGraphTensor* expo = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:k(expMask) name:nil];
        MPSGraphTensor* sub = [g equalWithPrimaryTensor:expo secondaryTensor:k(0) name:nil];
        bits = [g selectWithPredicateTensor:sub
                        truePredicateTensor:[g bitwiseANDWithPrimaryTensor:bits
                                                           secondaryTensor:k(signBit)
                                                                      name:nil]
                       falsePredicateTensor:bits
                                       name:nil];
    }
    MPSGraphTensor* mag = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:k(mask) name:nil];
    MPSGraphTensor* neg = [g lessThanWithPrimaryTensor:bits secondaryTensor:k(0) name:nil];
    MPSGraphTensor* key = [g selectWithPredicateTensor:neg
                                   truePredicateTensor:[g bitwiseXORWithPrimaryTensor:bits
                                                                      secondaryTensor:k(mask)
                                                                                 name:nil]
                                  falsePredicateTensor:bits
                                                  name:nil];
    MPSGraphTensor* isNaN = [g greaterThanWithPrimaryTensor:mag secondaryTensor:k(expMask) name:nil];
    return [g selectWithPredicateTensor:isNaN truePredicateTensor:k(ext) falsePredicateTensor:key name:nil];
}

// Inverse of FloatToOrderKey; the NaN key comes back as the canonical qNaN.
// `key` must not be rank 0 (reinterpretCastTensor aborts on rank 0).
static MPSGraphTensor* OrderKeyToFloat(MPSGraph* g, MPSGraphTensor* key, MPSDataType ft, bool isMax) {
    const bool wide = ft == MPSDataTypeFloat32;
    const MPSDataType it = wide ? MPSDataTypeInt32 : MPSDataTypeInt16;
    auto k = [&](int64_t v) { return OrderKeyConst(g, it, v); };
    const int64_t mask = wide ? 0x7FFFFFFF : 0x7FFF;
    const int64_t qnan = wide ? 0x7FC00000 : (ft == MPSDataTypeFloat16 ? 0x7E00 : 0x7FC0);
    const int64_t ext = isMax ? (wide ? INT32_MAX : INT16_MAX) : (wide ? INT32_MIN : INT16_MIN);
    MPSGraphTensor* keyNeg = [g lessThanWithPrimaryTensor:key secondaryTensor:k(0) name:nil];
    MPSGraphTensor* outBits = [g selectWithPredicateTensor:keyNeg
                                       truePredicateTensor:[g bitwiseXORWithPrimaryTensor:key
                                                                          secondaryTensor:k(mask)
                                                                                     name:nil]
                                      falsePredicateTensor:key
                                                      name:nil];
    MPSGraphTensor* outNaN = [g equalWithPrimaryTensor:key secondaryTensor:k(ext) name:nil];
    outBits = [g selectWithPredicateTensor:outNaN
                       truePredicateTensor:k(qnan)
                      falsePredicateTensor:outBits
                                      name:nil];
    return [g reinterpretCastTensor:outBits toType:ft name:nil];
}

// bf16 reduce_window max/min folds (general fold, bounded sliding fold, short
// scan) ran the WindowMaxMin tie rule at every step, which blew MPSGraph's
// fusion budget: [4096,4096] 2x2 pool 2.6-2.9x, 5-wide sliding min 16x. They
// fold the int16 order key instead (exact -0 < +0 and NaN at every step) and
// decode once. JAX_METALLIB_DISABLE_WINDOW_KEY_FOLD=1 keeps the per-step rule.
static bool UseWindowKeyFold(MPSGraphTensor* t) {
    static const bool disabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_WINDOW_KEY_FOLD");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    if (disabled || !WindowZeroTieEnabled() || t.dataType != MPSDataTypeBFloat16 || t.shape == nil ||
        t.shape.count == 0) {
        return false;
    }
    for (NSNumber* d in t.shape) {
        if (d.longLongValue <= 0) {
            return false;
        }
    }
    return true;
}

static MPSGraphTensor* KeyMaxMin(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool isMax) {
    return isMax ? [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil]
                 : [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* ReduceMaxMinViaKey(HandlerContext& ctx, MPSGraphTensor* input,
                                          NSArray<NSNumber*>* axes, const std::string& reductionType) {
    MPSGraph* g = ctx.graph;
    const MPSDataType ft = input.dataType;
    NSArray<NSNumber*>* shape = input.shape;
    if (!IsFloatForNaN(ft) || shape == nil || shape.count == 0 || axes.count == 0) {
        return nil;
    }
    for (NSNumber* d in shape) {
        if (d.longLongValue <= 0) {
            return nil;
        }
    }
    const bool isMax = reductionType == "stablehlo.maximum";
    const MPSDataType it = ft == MPSDataTypeFloat32 ? MPSDataTypeInt32 : MPSDataTypeInt16;
    MPSGraphTensor* key = FloatToOrderKey(g, input, isMax, /*flushF32Subnormal=*/true);

    MPSGraphTensor* red = TreeReduce(ctx, key, axes, reductionType);
    if (!red) {
        red = CreateReduction(g, key, axes, reductionType);
    }
    if (!red) {
        return nil;
    }
    if (red.dataType != it) {
        red = [g castTensor:red toType:it name:nil];
    }
    // reinterpretCastTensor of a rank-0 tensor aborts in the MPSGraph compiler;
    // the caller reshapes the result to the declared output shape.
    if (red.shape != nil && red.shape.count == 0) {
        red = [g reshapeTensor:red withShape:@[@1] name:nil];
    }
    return OrderKeyToFloat(g, red, ft, isMax);
}

// The key's reinterpretCast stops an elementwise producer from fusing into the
// reduce (f32 [4096,4096]: max(x*3+1, axis=1) 2.35x, maxabs_norm 1.70x). When
// the operand provably holds no -0 the plain reduce already orders every tie
// like XLA:CPU, so the key is skipped:
//  - kNonNegative (abs, exponential): never negative; f32 subnormal results are
//    flushed to +0 after the reduce, as XLA:CPU does.
//  - kNoZeroNoSubnormal (add/subtract with a splat constant c, |c| >=
//    2^(emin + p + 3)): an exact zero sum rounds to +0, and a nonzero result is
//    at least ~ulp(c)/4, so it is normal and no flush can produce -0.
// Function arguments, loop carries and every other producer keep the key.
// JAX_METALLIB_DISABLE_REDUCE_MAXMIN_PRODUCER_SKIP=1 always uses the key.
enum class MaxMinOperandClass { kUnknown, kNonNegative, kNoZeroNoSubnormal };

static bool ReduceMaxMinProducerSkipDisabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_REDUCE_MAXMIN_PRODUCER_SKIP");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return disabled;
}

// |c| of a non-NaN splat float constant (possibly behind broadcast_in_dim or
// reshape); false for anything else.
static bool SplatFloatMagnitude(mlir::Value v, double* mag) {
    for (int depth = 0; depth < 8 && v; ++depth) {
        mlir::Operation* def = v.getDefiningOp();
        if (!def) {
            return false;
        }
        if (auto c = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
            auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(c.getValue());
            if (!dense || !dense.isSplat()) {
                return false;
            }
            llvm::APFloat f = dense.getSplatValue<llvm::APFloat>();
            if (f.isNaN()) {
                return false;
            }
            bool losesInfo = false;
            f.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            *mag = std::fabs(f.convertToDouble());
            return true;
        }
        if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
            v = bc.getOperand();
            continue;
        }
        if (auto rs = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
            v = rs.getOperand();
            continue;
        }
        return false;
    }
    return false;
}

static MaxMinOperandClass ClassifyMaxMinOperand(mlir::Value v, MPSDataType ft) {
    mlir::Operation* def = v ? v.getDefiningOp() : nullptr;
    if (!def) {
        return MaxMinOperandClass::kUnknown;
    }
    if (mlir::isa<mlir::stablehlo::AbsOp, mlir::stablehlo::ExpOp>(def)) {
        return MaxMinOperandClass::kNonNegative;
    }
    if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp>(def)) {
        // 2^(emin + p + 3): f32 2^-99, f16 2^0, bf16 2^-115.
        const double minMag = ft == MPSDataTypeFloat32   ? std::ldexp(1.0, -99)
                              : ft == MPSDataTypeFloat16 ? 1.0
                                                         : std::ldexp(1.0, -115);
        for (mlir::Value operand : def->getOperands()) {
            double mag = 0.0;
            if (SplatFloatMagnitude(operand, &mag) && mag >= minMag) {
                return MaxMinOperandClass::kNoZeroNoSubnormal;
            }
        }
    }
    return MaxMinOperandClass::kUnknown;
}

// JAX_METALLIB_LEGACY_BITWISE_REDUCE=1 restores the MPSGraph reductionAnd/Or
// lowering (logical, not bitwise) and the "unsupported reduction type" error for xor.
static bool LegacyBitwiseReduce() {
    static const bool legacy = [] {
        const char* v = std::getenv("JAX_METALLIB_LEGACY_BITWISE_REDUCE");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return legacy;
}

static bool IsIntegerOrBoolType(MPSDataType t) {
    switch (t) {
    case MPSDataTypeBool:
    case MPSDataTypeInt8:
    case MPSDataTypeUInt8:
    case MPSDataTypeInt16:
    case MPSDataTypeUInt16:
    case MPSDataTypeInt32:
    case MPSDataTypeUInt32:
    case MPSDataTypeInt64:
    case MPSDataTypeUInt64:
        return true;
    default:
        return false;
    }
}

static MPSGraphTensor* BitwiseCombine(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b,
                                      const std::string& type) {
    if (type == "stablehlo.and") {
        return [g bitwiseANDWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    if (type == "stablehlo.or") {
        return [g bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    return [g bitwiseXORWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static int64_t IntegerBitWidth(MPSDataType t) {
    switch (t) {
    case MPSDataTypeInt8:
    case MPSDataTypeUInt8:
        return 8;
    case MPSDataTypeInt16:
    case MPSDataTypeUInt16:
        return 16;
    case MPSDataTypeInt64:
    case MPSDataTypeUInt64:
        return 64;
    default:
        return 32;
    }
}

// MPSGraph has no xor reduction, and its reductionAnd/reductionOr are logical: an
// integer and/or reduce returned 0/1 (uint8 [5,7,3] axis 1: 15/15 wrong) and a bool
// and-reduce over a middle axis was wrong on 6/15 slices. The init operand was also
// ignored. Lower every integer/bool and/or/xor reduce exactly instead: move the
// reduced axes last and flatten them to one axis of length L, then
//  1. pairwise-fold the two halves with the elementwise op for at most 8 levels (an
//     odd tail element is carried). An uncapped fold is a reconvergent chain whose
//     MPSGraph first-call cost grows with depth: depth 20 ~1 GB / 4-6 s, depth 22
//     2.6 GB / 20 s, and a 16M-element jnp.all passed the 8 GB memory guard;
//  2. unpack what is left to [K, n, W] bits ((x >> b) & 1, exact for signed types),
//     reduce over n natively (xor: int32 sum & 1, or: uint8 max, and: uint8 min),
//     shift the bits back and OR-fold the W axis (depth <= 6);
//  3. combine with init.
// Bool is one 0/1 uint8 plane reduced natively over the original axes, no fold. u32 full reduce of 2^24
// elements: ~0.15 s first call, 3.3-4.0 ms (emulated in build/probe_r20i_hybrid.log).
static ProcessResult ReduceBitwiseFold(HandlerContext& ctx, MPSGraphTensor* input,
                                       const std::vector<int64_t>& dims, const std::string& type) {
    MPSGraph* g = ctx.graph;
    MPSGraphTensor* init = GetInputTensor(ctx, 1);
    NSArray<NSNumber*>* shape = input.shape;
    NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
    if (!init || !shape || !outShape) {
        return ProcessResult::Error("reduce: bitwise fold needs init and static shapes");
    }
    const bool isBool = input.dataType == MPSDataTypeBool;
    MPSGraphTensor* x = isBool ? [g castTensor:input toType:MPSDataTypeUInt8 name:nil] : input;
    if (init.dataType != x.dataType) {
        init = [g castTensor:init toType:x.dataType name:nil];
    }
    const NSUInteger rank = shape.count;
    std::vector<bool> reduced(rank, false);
    for (int64_t d : dims) {
        if (d >= 0 && (NSUInteger)d < rank) {
            reduced[(size_t)d] = true;
        }
    }
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    int64_t K = 1;
    int64_t L = 1;
    for (NSUInteger i = 0; i < rank; ++i) {
        if (!reduced[i]) {
            [perm addObject:@(i)];
            K *= shape[i].longLongValue;
        }
    }
    for (NSUInteger i = 0; i < rank; ++i) {
        if (reduced[i]) {
            [perm addObject:@(i)];
            L *= shape[i].longLongValue;
        }
    }
    MPSGraphTensor* result = nil;
    if (K == 0) {
        result = [g broadcastTensor:init toShape:outShape name:nil];
    } else if (isBool) {
        // One 0/1 plane: reduce it natively over the original axes (no transpose;
        // moving axis 0 last cost a full copy, jnp.any(axis=0) 1.94x the old path).
        NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
        for (NSUInteger i = 0; i < rank; ++i) {
            if (reduced[i]) {
                [axes addObject:@(i)];
            }
        }
        MPSGraphTensor* r = nil;
        if (type == "stablehlo.xor") {
            r = [g castTensor:x toType:MPSDataTypeInt32 name:nil];
            r = [g reductionSumWithTensor:r axes:axes name:nil];
            MPSGraphTensor* oneI = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
            r = [g bitwiseANDWithPrimaryTensor:r secondaryTensor:oneI name:nil];
            r = [g castTensor:r toType:MPSDataTypeUInt8 name:nil];
        } else {
            r = type == "stablehlo.or" ? [g reductionMaximumWithTensor:x axes:axes name:nil]
                                       : [g reductionMinimumWithTensor:x axes:axes name:nil];
        }
        result = BitwiseCombine(g, r, init, type);
        result = [g reshapeTensor:result withShape:outShape name:nil];
    } else {
        bool identityPerm = true;
        for (NSUInteger i = 0; i < rank; ++i) {
            identityPerm = identityPerm && perm[i].unsignedIntegerValue == i;
        }
        MPSGraphTensor* cur = x;
        if (!identityPerm) {
            cur = [g transposeTensor:cur permutation:perm name:nil];
        }
        cur = [g reshapeTensor:cur withShape:@[@(K), @(L)] name:nil];
        const int64_t width = isBool ? 1 : IntegerBitWidth(x.dataType);
        const int maxDepth = isBool ? 0 : 8;
        int64_t len = L;
        for (int depth = 0; len > 1 && depth < maxDepth; ++depth) {
            const int64_t h = len / 2;
            MPSGraphTensor* a = [g sliceTensor:cur
                                        starts:@[@0, @0]
                                          ends:@[@(K), @(h)]
                                       strides:@[@1, @1]
                                          name:nil];
            MPSGraphTensor* b = [g sliceTensor:cur
                                        starts:@[@0, @(h)]
                                          ends:@[@(K), @(2 * h)]
                                       strides:@[@1, @1]
                                          name:nil];
            MPSGraphTensor* r = BitwiseCombine(g, a, b, type);
            if (len & 1) {
                MPSGraphTensor* tail = [g sliceTensor:cur
                                               starts:@[@0, @(2 * h)]
                                                 ends:@[@(K), @(len)]
                                              strides:@[@1, @1]
                                                 name:nil];
                r = [g concatTensors:@[r, tail] dimension:1 name:nil];
            }
            cur = r;
            len = h + (len & 1);
        }
        if (len > 1) {
            std::vector<int32_t> shiftVals((size_t)width);
            for (int64_t b = 0; b < width; ++b) {
                shiftVals[(size_t)b] = (int32_t)b;
            }
            MPSGraphTensor* shifts = [g constantWithData:[NSData dataWithBytes:shiftVals.data()
                                                                        length:shiftVals.size() * 4]
                                                   shape:@[@1, @1, @(width)]
                                                dataType:MPSDataTypeInt32];
            if (cur.dataType != MPSDataTypeInt32) {
                shifts = [g castTensor:shifts toType:cur.dataType name:nil];
            }
            MPSGraphTensor* one = [g constantWithScalar:1 dataType:cur.dataType];
            MPSGraphTensor* e = [g reshapeTensor:cur withShape:@[@(K), @(len), @1] name:nil];
            MPSGraphTensor* bits = [g bitwiseRightShiftWithPrimaryTensor:e secondaryTensor:shifts name:nil];
            bits = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:one name:nil];
            MPSGraphTensor* r = nil;
            if (type == "stablehlo.xor") {
                r = [g castTensor:bits toType:MPSDataTypeInt32 name:nil];
                r = [g reductionSumWithTensor:r axes:@[@1] name:nil];
                MPSGraphTensor* oneI = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
                r = [g bitwiseANDWithPrimaryTensor:r secondaryTensor:oneI name:nil];
            } else {
                r = [g castTensor:bits toType:MPSDataTypeUInt8 name:nil];
                r = type == "stablehlo.or" ? [g reductionMaximumWithTensor:r axes:@[@1] name:nil]
                                           : [g reductionMinimumWithTensor:r axes:@[@1] name:nil];
            }
            if (r.dataType != cur.dataType) {
                r = [g castTensor:r toType:cur.dataType name:nil];
            }
            r = [g bitwiseLeftShiftWithPrimaryTensor:r secondaryTensor:shifts name:nil];
            cur = [g reshapeTensor:r withShape:@[@(K), @(width)] name:nil];
            for (int64_t w = width; w > 1; w /= 2) {
                MPSGraphTensor* lo = [g sliceTensor:cur
                                             starts:@[@0, @0]
                                               ends:@[@(K), @(w / 2)]
                                            strides:@[@1, @1]
                                               name:nil];
                MPSGraphTensor* hi = [g sliceTensor:cur
                                             starts:@[@0, @(w / 2)]
                                               ends:@[@(K), @(w)]
                                            strides:@[@1, @1]
                                               name:nil];
                cur = [g bitwiseORWithPrimaryTensor:lo secondaryTensor:hi name:nil];
            }
        }
        result = BitwiseCombine(g, cur, init, type);
        result = [g reshapeTensor:result withShape:outShape name:nil];
    }
    if (isBool) {
        result = [g castTensor:result toType:MPSDataTypeBool name:nil];
    }
    return Result(ctx, result, "reduce_bitwise_fold");
}

// JAX_METALLIB_DISABLE_REDUCE_INIT_COMBINE=1 drops a non-identity reduce init
// again (the pre-fix behaviour: lax.reduce(x, 5., add) returned the plain sum).
static bool ReduceInitCombineDisabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_REDUCE_INIT_COMBINE");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return disabled;
}

// Scalar constant init that cannot change an add/mul/max/min reduction, so the
// combine step is skipped and jnp reductions keep their lowering bit for bit.
// add: any zero (as in reduce_window), mul: one, max: -inf / type minimum,
// min: +inf / type maximum.
static bool IsReduceIdentityInit(mlir::Value init, const std::string& type) {
    mlir::DenseElementsAttr dense;
    if (!GetScalarConstantAttr(init, dense)) {
        return false;
    }
    if (type == "stablehlo.add") {
        return IsScalarZero(init);
    }
    const bool isMax = type == "stablehlo.maximum";
    const bool isMin = type == "stablehlo.minimum";
    auto elemType = dense.getElementType();
    if (mlir::isa<mlir::FloatType>(elemType)) {
        llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
        if (type == "stablehlo.multiply") {
            return ap.isExactlyValue(1.0);
        }
        return (isMax || isMin) && ap.isInfinity() && ap.isNegative() == isMax;
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        llvm::APInt ap = *dense.getValues<llvm::APInt>().begin();
        if (type == "stablehlo.multiply") {
            return ap.isOne();
        }
        const bool isUnsigned = intType.isUnsigned() || ap.getBitWidth() == 1;
        if (isMax) {
            return isUnsigned ? ap.isZero() : ap.isMinSignedValue();
        }
        if (isMin) {
            return isUnsigned ? ap.isMaxValue() : ap.isMaxSignedValue();
        }
    }
    return false;
}

// The window helper's integer path also normalizes reduce init bits without
// changing the float-select default used by existing window lowerings.
static MPSGraphTensor* FlushF32SubnormalSigned(MPSGraph* g, MPSGraphTensor* t, bool forceBits = false);

// IEEE add of exclusively -0 leaves stays -0 for every reduction tree. The
// graph sum can start at +0 instead; correct only this exact value invariant,
// not cancellation, subnormals or a CPU compiler's shape-dependent YNN path.
static bool CorrectF32AddNegativeZero(HandlerContext& ctx, MPSGraphTensor*& result, MPSGraphTensor* init) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return false;
    }
    auto dimensions = mlir::cast<mlir::stablehlo::ReduceOp>(ctx.op).getDimensions();
    if (input.shape == nil || result.shape == nil) {
        return true;
    }
    // Empty reduced axes already return the init directly; empty output tensors
    // need no correction and must not enter an MPSGraph reduction encode.
    for (NSNumber* dim in input.shape) {
        if (dim.longLongValue == 0) {
            return true;
        }
    }
    MPSGraph* g = ctx.graph;
    NSArray<NSNumber*>* shape = result.shape;
    auto bits = [&](MPSGraphTensor* t) {
        if (t.shape.count == 0) {
            t = [g reshapeTensor:t withShape:@[@1] name:nil];
        }
        return [g reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
    };
    MPSGraphTensor* negZero = OrderKeyConst(g, MPSDataTypeInt32, INT32_MIN);
    MPSGraphTensor* other = [g notEqualWithPrimaryTensor:bits(input) secondaryTensor:negZero name:nil];
    other = [g castTensor:other toType:MPSDataTypeUInt8 name:nil];
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }
    // Bool/bitwise reductions are not reliable here; native uint8 max over a
    // 0/1 mask says whether any input is not exactly -0. With no axes, keep
    // the per-element mask and do not encode an empty-axis reduction.
    MPSGraphTensor* anyOther =
        dimensions.empty() ? other : [g reductionMaximumWithTensor:other axes:axes name:nil];
    NSArray<NSNumber*>* bitShape = shape.count == 0 ? @[@1] : shape;
    if (![bitShape isEqualToArray:anyOther.shape]) {
        anyOther = [g reshapeTensor:anyOther withShape:bitShape name:nil];
    }
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeUInt8];
    MPSGraphTensor* allNegZero = [g equalWithPrimaryTensor:anyOther secondaryTensor:zero name:nil];
    MPSGraphTensor* initNegZero = [g equalWithPrimaryTensor:bits(init) secondaryTensor:negZero name:nil];
    MPSGraphTensor* predicate = [g logicalANDWithPrimaryTensor:allNegZero
                                               secondaryTensor:initNegZero
                                                          name:nil];
    MPSGraphTensor* selected = [g selectWithPredicateTensor:predicate
                                        truePredicateTensor:negZero
                                       falsePredicateTensor:bits(result)
                                                       name:nil];
    result = [g reinterpretCastTensor:selected toType:MPSDataTypeFloat32 name:nil];
    if (![shape isEqualToArray:result.shape]) {
        result = [g reshapeTensor:result withShape:shape name:nil];
    }
    return true;
}

// XLA folds the init into every reduced slice; MPSGraph reductions have no init,
// so a non-identity init (lax.reduce(x, 5., add), a traced init, a max init that
// beats the data) was silently dropped (build/probe_r20k_reduce_init*.log: every
// such add/mul/max/min case returned the init-free result). Combine it after the
// reduction; max/min go through WindowMaxMin for NaN propagation and -0 < +0.
static bool CombineReduceInit(HandlerContext& ctx, MPSGraphTensor*& result, const std::string& type) {
    mlir::Value initValue = ctx.op->getOperand(1);
    if (!result || ReduceInitCombineDisabled()) {
        return true;
    }
    bool correctAddZero = type == "stablehlo.add" && result.dataType == MPSDataTypeFloat32 &&
                          mlir::cast<mlir::ShapedType>(initValue.getType()).getElementType().isF32();
    mlir::DenseElementsAttr dense;
    if (correctAddZero && GetScalarConstantAttr(initValue, dense)) {
        llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
        correctAddZero = ap.isZero() && ap.isNegative();
    }
    const bool identity = IsReduceIdentityInit(initValue, type);
    if (identity && !correctAddZero) {
        return true;
    }
    MPSGraphTensor* init = GetInputTensor(ctx, 1);
    if (!init) {
        return false;
    }
    if (init.dataType != result.dataType) {
        init = [ctx.graph castTensor:init toType:result.dataType name:nil];
    }
    if (type == "stablehlo.add") {
        if (!identity) {
            result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
        }
        if (correctAddZero && !CorrectF32AddNegativeZero(ctx, result, init)) {
            return false;
        }
    } else if (type == "stablehlo.multiply") {
        result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:init name:nil];
    } else if (type == "stablehlo.maximum" || type == "stablehlo.minimum") {
        const bool flushF32Subnormal = init.dataType == MPSDataTypeFloat32;
        if (flushF32Subnormal) {
            init = FlushF32SubnormalSigned(ctx.graph, init, /*forceBits=*/true);
        }
        result =
            WindowMaxMin(ctx.graph, result, init, type == "stablehlo.maximum", initValue, flushF32Subnormal);
    }
    return true;
}

static ProcessResult HandleSingleResultReduce(HandlerContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(ctx.op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("reduce: input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    // Reducing a zero-length axis aborts inside MPSGraph's reduction encode (f32/i32
    // sum over [0] or [4,0]: SIGABRT). XLA defines the result as the init value
    // broadcast to the output shape, so emit exactly that and skip the reduction.
    NSArray<NSNumber*>* inputShape = input.shape;
    bool emptyReducedAxis = false;
    for (int64_t dim : dimensions) {
        if (inputShape && dim >= 0 && (NSUInteger)dim < inputShape.count &&
            inputShape[(NSUInteger)dim].longLongValue == 0) {
            emptyReducedAxis = true;
        }
    }
    if (emptyReducedAxis) {
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return ProcessResult::Error("reduce: init value tensor not found");
        }
        MPSDataType outType = GetResultMpsType(ctx.op, 0);
        if (outType != MPSDataTypeInvalid && init.dataType != outType) {
            init = [ctx.graph castTensor:init toType:outType name:nil];
        }
        NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
        MPSGraphTensor* filled = init;
        if (outShape && outShape.count > 0) {
            filled = [ctx.graph broadcastTensor:init toShape:outShape name:nil];
        } else if (outShape && init.shape && init.shape.count > 0) {
            filled = [ctx.graph reshapeTensor:init withShape:outShape name:nil];
        }
        return Result(ctx, filled, "reduce_empty_axis");
    }

    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    if ((reductionType == "stablehlo.and" || reductionType == "stablehlo.or" ||
         reductionType == "stablehlo.xor") &&
        IsIntegerOrBoolType(input.dataType) && !LegacyBitwiseReduce()) {
        std::vector<int64_t> dimVec(dimensions.begin(), dimensions.end());
        return ReduceBitwiseFold(ctx, input, dimVec, reductionType);
    }

    bool flushF32Subnormal = false;
    if ((reductionType == "stablehlo.maximum" || reductionType == "stablehlo.minimum") &&
        IsFloatForNaN(input.dataType) && !ReduceMaxMinKeyDisabled()) {
        const MaxMinOperandClass cls = ReduceMaxMinProducerSkipDisabled()
                                           ? MaxMinOperandClass::kUnknown
                                           : ClassifyMaxMinOperand(ctx.op->getOperand(0), input.dataType);
        flushF32Subnormal = cls == MaxMinOperandClass::kNonNegative && input.dataType == MPSDataTypeFloat32;
        MPSGraphTensor* keyed =
            cls == MaxMinOperandClass::kUnknown ? ReduceMaxMinViaKey(ctx, input, axes, reductionType) : nil;
        if (keyed) {
            if (!CombineReduceInit(ctx, keyed, reductionType)) {
                return ProcessResult::Error("reduce: init value tensor not found");
            }
            NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
            if (outShape && keyed.shape && ![outShape isEqualToArray:keyed.shape]) {
                keyed = [ctx.graph reshapeTensor:keyed withShape:outShape name:nil];
            }
            return Result(ctx, keyed, "reduce_maxmin_key");
        }
    }

    int64_t numReduceDims = (int64_t)dimensions.size();
    MPSGraphTensor* treeResult = TreeReduce(ctx, input, axes, reductionType);

    MPSGraphTensor* result = nullptr;
    if (treeResult) {
        result = treeResult;
    } else {
        if (reductionType == "stablehlo.add") {
            result = [ctx.graph reductionSumWithTensor:input axes:axes name:nil];
        } else if (reductionType == "stablehlo.multiply") {
            result = [ctx.graph reductionProductWithTensor:input axes:axes name:nil];
        } else if (reductionType == "stablehlo.maximum" || reductionType == "stablehlo.minimum") {
            result = CreateReduction(ctx.graph, input, axes, reductionType);
        } else if (reductionType == "stablehlo.and") {
            result = [ctx.graph reductionAndWithTensor:input axes:axes name:nil];
        } else if (reductionType == "stablehlo.or") {
            result = [ctx.graph reductionOrWithTensor:input axes:axes name:nil];
        } else {
            return ProcessResult::Error("reduce: unsupported reduction type: " + reductionType);
        }
    }

    if (flushF32Subnormal && result && result.dataType == MPSDataTypeFloat32) {
        // Non-negative operand: only a positive subnormal can differ from XLA:CPU,
        // which flushes it to +0. NaN compares false and is kept.
        MPSGraphTensor* tiny = [ctx.graph constantWithScalar:std::ldexp(1.0, -126)
                                                    dataType:MPSDataTypeFloat32];
        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* small = [ctx.graph lessThanWithPrimaryTensor:result secondaryTensor:tiny name:nil];
        result = [ctx.graph selectWithPredicateTensor:small
                                  truePredicateTensor:zero
                                 falsePredicateTensor:result
                                                 name:nil];
    }

    if (!CombineReduceInit(ctx, result, reductionType)) {
        return ProcessResult::Error("reduce: init value tensor not found");
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && result && result.shape && ![outputShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }

    return Result(ctx, result, "reduce");
}

// The arg-reduction finds the best input pair, but MPSGraph has no init operands.
// Fold the init pair through the actual StableHLO body so its independent value
// and index selects (including ties and NaNs) are preserved. JAX's constant
// extremum sentinel + index zero cannot win, so its common lowering stays intact.
static ProcessResult CombineArgReduceInit(HandlerContext& ctx, mlir::stablehlo::ReduceOp reduceOp,
                                          ArgReduceKind kind, MPSGraphTensor*& valueOut,
                                          MPSGraphTensor*& indexOut) {
    auto inits = reduceOp.getInitValues();
    const std::string type = kind == ArgReduceKind::kMax ? "stablehlo.maximum" : "stablehlo.minimum";
    if (ReduceInitCombineDisabled() || (IsReduceIdentityInit(inits[0], type) && IsScalarZero(inits[1]))) {
        return ProcessResult{};
    }
    if (!ctx.processBlock) {
        return ProcessResult::Error("reduce: missing block processor for arg-reduce init");
    }
    MPSGraphTensor* valueInit = GetInputTensor(ctx, 2);
    MPSGraphTensor* indexInit = GetInputTensor(ctx, 3);
    if (!valueInit || !indexInit) {
        return ProcessResult::Error("reduce: arg-reduce init tensor not found");
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() != 4) {
        return ProcessResult::Error("reduce: arg-reduce body argument arity mismatch");
    }
    ValueMap bodyValues = ctx.values;
    bodyValues[body.getArgument(0).getAsOpaquePointer()] = valueInit;
    bodyValues[body.getArgument(1).getAsOpaquePointer()] = indexInit;
    bodyValues[body.getArgument(2).getAsOpaquePointer()] = valueOut;
    bodyValues[body.getArgument(3).getAsOpaquePointer()] = indexOut;
    HandlerContext bodyCtx(ctx.graph, nullptr, bodyValues, ctx.module, ctx.depth + 1, ctx.processBlock);
    bodyCtx.variables = ctx.variables;
    bodyCtx.in_control_flow = ctx.isInControlFlow();
    ProcessResult combined = ctx.processBlock(bodyCtx, body);
    if (!combined.ok()) {
        return combined;
    }
    if (combined.return_values.size() != 2) {
        return ProcessResult::Error("reduce: arg-reduce body must return two values");
    }
    valueOut = GetTensor(bodyValues, combined.return_values[0]);
    indexOut = GetTensor(bodyValues, combined.return_values[1]);
    if (!valueOut || !indexOut) {
        return ProcessResult::Error("reduce: combined arg-reduce result tensor not found");
    }
    // MPSGraph rewrote a +0 gathered value to a constant -0 on this float
    // select, even though the predicate chose the gathered value (traced
    // inits and integer-bits selects were exact). Preserve the selected bits.
    // Only the constant-zero f32 case needs this extra graph barrier.
    auto selectOp = combined.return_values[0].getDefiningOp<mlir::stablehlo::SelectOp>();
    if (valueInit.dataType == MPSDataTypeFloat32 && IsScalarZero(inits[0]) && selectOp) {
        MPSGraphTensor* pred = GetTensor(bodyValues, selectOp->getOperand(0));
        auto bits = [&](mlir::Value v) {
            MPSGraphTensor* t = GetTensor(bodyValues, v);
            if (t.shape.count == 0) {
                t = [ctx.graph reshapeTensor:t withShape:@[@1] name:nil];
            }
            return [ctx.graph reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
        };
        MPSGraphTensor* selected = [ctx.graph selectWithPredicateTensor:pred
                                                    truePredicateTensor:bits(selectOp->getOperand(1))
                                                   falsePredicateTensor:bits(selectOp->getOperand(2))
                                                                   name:nil];
        valueOut = [ctx.graph reinterpretCastTensor:selected toType:MPSDataTypeFloat32 name:nil];
    }
    return ProcessResult{};
}

static ProcessResult HandleMultiResultReduce(HandlerContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(ctx.op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }
    if (ctx.op->getNumResults() != 2 || ctx.op->getNumOperands() != 4) {
        return ProcessResult::Error("reduce: unsupported multi-result shape");
    }

    MPSGraphTensor* valueInput = GetInputTensor(ctx, 0);
    if (!valueInput) {
        return ProcessResult::Error("reduce: value input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1) {
        return ProcessResult::Error("reduce: only single-axis multi-result reduce is supported");
    }
    NSInteger axis = (NSInteger)dimensions[0];

    ArgReduceKind kind = detectArgReduceKind(reduceOp);
    if (kind == ArgReduceKind::kUnknown) {
        return ProcessResult::Error("reduce: unsupported multi-result reduce body");
    }

    MPSGraph* g = ctx.graph;
    // MPSGraph arg-reductions over a non-last axis were far slower than over the
    // last axis, so that axis is moved to the end (all dtypes). Float inputs then
    // arg-reduce an int32 order key (below) when the axis was moved or rows are
    // <= kArgKeyMaxRow, replacing the max + arg-max + NaN pass. Measured argmax:
    // f32 [4096,4096] axis 0 5.19 -> 1.87 ms, [64,512,512] axis 0 30.9 -> 3.3 ms,
    // i32 [256,65536] axis 0 8.45 -> 2.05 ms, f32 [65536,256] axis 1 4.51 -> 3.11
    // ms, [16384,2048] axis 1 5.91 -> 4.57 ms. At 4096 columns the key tied (f32)
    // or lost (f16 [4096,4096] 1.25 -> 1.55 ms), hence the row limit.
    // JAX_METALLIB_DISABLE_ARGREDUCE_KEY=1 restores the direct lowering.
    static const bool keyDisabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_ARGREDUCE_KEY");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    constexpr int64_t kArgKeyMaxRow = 2048;
    const NSUInteger rank = valueInput.shape.count;
    bool movedAxis = false;
    if (!keyDisabled && rank > 1 && axis != (NSInteger)rank - 1) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger d = 0; d < rank; ++d) {
            if ((NSInteger)d != axis) {
                [perm addObject:@(d)];
            }
        }
        [perm addObject:@(axis)];
        valueInput = [g transposeTensor:valueInput permutation:perm name:nil];
        axis = (NSInteger)rank - 1;
        movedAxis = true;
    }
    const int64_t rowLen = rank > 0 ? [valueInput.shape[axis] longLongValue] : 1;
    const bool useKey = !keyDisabled && IsFloatForNaN(valueInput.dataType) &&
                        (movedAxis || (rowLen >= 0 && rowLen <= kArgKeyMaxRow));

    // MPSNDArrayReduction asserts (aborting the process) on uint32 arg-reductions.
    // XOR with the sign bit maps uint32 order onto int32 order; reduce there and
    // map the value back.
    const bool uint32Input = valueInput.dataType == MPSDataTypeUInt32;
    MPSGraphTensor* signBit = nil;
    if (uint32Input) {
        signBit = [ctx.graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
        valueInput = [ctx.graph reinterpretCastTensor:valueInput toType:MPSDataTypeInt32 name:nil];
        valueInput = [ctx.graph bitwiseXORWithPrimaryTensor:valueInput secondaryTensor:signBit name:nil];
    }

    MPSGraphTensor* valueOut = nullptr;
    MPSGraphTensor* indexOut = nullptr;
    if (useKey) {
        // Order-preserving int32 key from the bit pattern: x >= 0 keeps its bits,
        // x < 0 becomes -(bits & mask) (so -0 and +0 tie at 0), NaN becomes
        // INT32_MAX for argmax / INT32_MIN for argmin. One integer arg-reduction
        // then yields XLA's index (first NaN, else first extremum) without a
        // separate NaN pass, and the bit test does not depend on fast-math isNaN.
        auto i32 = [&](int32_t v) {
            return [g constantWithData:[NSData dataWithBytes:&v length:sizeof(v)]
                                 shape:@[@1]
                              dataType:MPSDataTypeInt32];
        };
        const MPSDataType ft = valueInput.dataType;
        const bool wide = ft == MPSDataTypeFloat32;
        const MPSDataType bitsType = wide ? MPSDataTypeInt32 : MPSDataTypeInt16;
        MPSGraphTensor* bits = [g reinterpretCastTensor:valueInput toType:bitsType name:nil];
        if (!wide) {
            bits = [g castTensor:bits toType:MPSDataTypeInt32 name:nil]; // sign-extends
        }
        const int32_t magMask = wide ? 0x7FFFFFFF : 0x7FFF;
        const int32_t infBits = wide ? 0x7F800000 : (ft == MPSDataTypeFloat16 ? 0x7C00 : 0x7F80);
        MPSGraphTensor* mag = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:i32(magMask) name:nil];
        MPSGraphTensor* neg = [g lessThanWithPrimaryTensor:bits secondaryTensor:i32(0) name:nil];
        MPSGraphTensor* key = [g selectWithPredicateTensor:neg
                                       truePredicateTensor:[g negativeWithTensor:mag name:nil]
                                      falsePredicateTensor:mag
                                                      name:nil];
        MPSGraphTensor* isNaN = [g greaterThanWithPrimaryTensor:mag secondaryTensor:i32(infBits) name:nil];
        MPSGraphTensor* nanKey = i32(kind == ArgReduceKind::kMax ? INT32_MAX : INT32_MIN);
        key = [g selectWithPredicateTensor:isNaN
                       truePredicateTensor:nanKey
                      falsePredicateTensor:key
                                      name:nil];
        indexOut = kind == ArgReduceKind::kMax ? [g reductionArgMaximumWithTensor:key axis:axis name:nil]
                                               : [g reductionArgMinimumWithTensor:key axis:axis name:nil];
    } else if (kind == ArgReduceKind::kMax) {
        valueOut = [ctx.graph reductionMaximumWithTensor:valueInput axis:axis name:nil];
        indexOut = [ctx.graph reductionArgMaximumWithTensor:valueInput axis:axis name:nil];
    } else {
        valueOut = [ctx.graph reductionMinimumWithTensor:valueInput axis:axis name:nil];
        indexOut = [ctx.graph reductionArgMinimumWithTensor:valueInput axis:axis name:nil];
    }
    if ((!useKey && !valueOut) || !indexOut) {
        return ProcessResult::Error("reduce: failed to lower multi-result reduce");
    }
    if (uint32Input) {
        valueOut = [ctx.graph bitwiseXORWithPrimaryTensor:valueOut secondaryTensor:signBit name:nil];
        valueOut = [ctx.graph reinterpretCastTensor:valueOut toType:MPSDataTypeUInt32 name:nil];
    }
    // MPSGraph arg-reductions returned -1 when every element of the slice equals
    // the reduction sentinel (e.g. argmin over an all-INT32_MAX column). That only
    // happens when all elements are equal, where the first index (0) is correct.
    MPSGraphTensor* zeroIndex = [ctx.graph constantWithScalar:0 dataType:indexOut.dataType];
    indexOut = [ctx.graph maximumWithPrimaryTensor:indexOut secondaryTensor:zeroIndex name:nil];
    if (useKey) {
        MPSGraphTensor* gatherIdx = indexOut.dataType == MPSDataTypeInt32
                                        ? indexOut
                                        : [g castTensor:indexOut toType:MPSDataTypeInt32 name:nil];
        valueOut = [g gatherAlongAxis:axis withUpdatesTensor:valueInput indicesTensor:gatherIdx name:nil];
    }

    // Wide float rows (no int32 key): the uint8 NaN-mask pass below cost more than
    // the arg-reduction itself (f32 [4096,4096] argmax 1.78 ms, plain int32 argmax
    // 0.74 ms). A NaN-propagating row extremum decides whether any row has NaN and
    // an MPSGraph `if` runs the mask pass only then; the value is gathered at the
    // final index (XLA returns the element at that index). Top-level graphs only.
    // Measured, NaN-free rows: f32 [4096,4096] 1.78-2.05 -> 1.31-1.36 ms, f16
    // 1.21-1.25 -> 0.84-1.01 ms. Inputs with a NaN row pay the extra pass (f32
    // [4096,4096] 1.77-1.82 -> 2.34-2.75 ms).
    // JAX_METALLIB_DISABLE_ARGREDUCE_NAN_IF=1 always runs the mask pass.
    static const bool nanIfDisabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_ARGREDUCE_NAN_IF");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    bool nanHandled = false;
    if (!useKey && !nanIfDisabled && !LegacyArgReduceNaN() && g_control_flow_depth == 0 &&
        IsFloatForNaN(valueInput.dataType)) {
        const bool isMax = kind == ArgReduceKind::kMax;
        MPSGraphTensor* rowExt =
            isMax ? [g reductionMaximumPropagateNaNWithTensor:valueInput axis:axis name:nil]
                  : [g reductionMinimumPropagateNaNWithTensor:valueInput axis:axis name:nil];
        MPSGraphTensor* rowNaN = [g isNaNWithTensor:rowExt name:nil];
        MPSGraphTensor* anyU8 = [g castTensor:rowNaN toType:MPSDataTypeUInt8 name:nil];
        anyU8 = [g reshapeTensor:anyU8 withShape:@[@-1] name:nil];
        anyU8 = [g reductionMaximumWithTensor:anyU8 axis:0 name:nil];
        MPSGraphTensor* zeroU8 = [g constantWithScalar:0 dataType:MPSDataTypeUInt8];
        MPSGraphTensor* pred = [g notEqualWithPrimaryTensor:anyU8 secondaryTensor:zeroU8 name:nil];
        pred = [g reshapeTensor:pred withShape:@[] name:nil];
        MPSGraphTensor* baseIndex = indexOut.dataType == MPSDataTypeInt32
                                        ? indexOut
                                        : [g castTensor:indexOut toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* input = valueInput;
        const NSInteger reduceAxis = axis;
        NSArray<MPSGraphTensor*>* outs = [g ifWithPredicateTensor:pred
            thenBlock:^NSArray<MPSGraphTensor*>*() {
              MPSGraphTensor* mask = [g castTensor:[g isNaNWithTensor:input name:nil]
                                            toType:MPSDataTypeUInt8
                                              name:nil];
              MPSGraphTensor* nanIdx = [g reductionArgMaximumWithTensor:mask axis:reduceAxis name:nil];
              if (nanIdx.dataType != MPSDataTypeInt32) {
                  nanIdx = [g castTensor:nanIdx toType:MPSDataTypeInt32 name:nil];
              }
              MPSGraphTensor* zeroI = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
              nanIdx = [g maximumWithPrimaryTensor:nanIdx secondaryTensor:zeroI name:nil];
              return @[[g selectWithPredicateTensor:rowNaN
                                truePredicateTensor:nanIdx
                               falsePredicateTensor:baseIndex
                                               name:nil]];
            }
            elseBlock:^NSArray<MPSGraphTensor*>*() {
              return @[[g identityWithTensor:baseIndex name:nil]];
            }
            name:nil];
        if (!outs || outs.count != 1) {
            return ProcessResult::Error("reduce: arg-reduce NaN if returned wrong arity");
        }
        indexOut = outs[0];
        valueOut = [g gatherAlongAxis:axis withUpdatesTensor:valueInput indicesTensor:indexOut name:nil];
        nanHandled = true;
    }

    // JAX argmax/argmin return the first NaN's index and a NaN value when the
    // slice contains NaN; MPSGraph arg-reductions ignored NaN entirely.
    if (!useKey && !nanHandled && IsFloatForNaN(valueInput.dataType)) {
        MPSGraphTensor* nanMask = [ctx.graph isNaNWithTensor:valueInput name:nil];
        MPSGraphTensor* nanIndex = nil;
        MPSGraphTensor* anyNaN = nil;
        MPSGraphTensor* zeroI = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
        if (LegacyArgReduceNaN()) {
            MPSGraphTensor* nanInt = [ctx.graph castTensor:nanMask toType:MPSDataTypeInt32 name:nil];
            nanIndex = [ctx.graph reductionArgMaximumWithTensor:nanInt axis:axis name:nil];
            anyNaN = [ctx.graph reductionMaximumWithTensor:nanInt axis:axis name:nil];
            anyNaN = [ctx.graph greaterThanWithPrimaryTensor:anyNaN secondaryTensor:zeroI name:nil];
        } else {
            // One narrow pass instead of an int32 mask plus two full reductions
            // (f32 [4096,4096] argmax 3.3 ms, of which the plain arg-reduction is
            // 0.84 ms): arg-max a uint8 mask for the first NaN, then "any NaN" is
            // whether the element at that index is NaN. MPSGraph returns -1 on an
            // all-equal slice (no NaN), so clamp to 0 before the gather.
            MPSGraphTensor* nanU8 = [ctx.graph castTensor:nanMask toType:MPSDataTypeUInt8 name:nil];
            nanIndex = [ctx.graph reductionArgMaximumWithTensor:nanU8 axis:axis name:nil];
            if (nanIndex.dataType != MPSDataTypeInt32) {
                nanIndex = [ctx.graph castTensor:nanIndex toType:MPSDataTypeInt32 name:nil];
            }
            nanIndex = [ctx.graph maximumWithPrimaryTensor:nanIndex secondaryTensor:zeroI name:nil];
            MPSGraphTensor* hit = [ctx.graph gatherAlongAxis:axis
                                           withUpdatesTensor:valueInput
                                               indicesTensor:nanIndex
                                                        name:nil];
            anyNaN = [ctx.graph isNaNWithTensor:hit name:nil];
        }
        if (nanIndex.dataType != indexOut.dataType) {
            nanIndex = [ctx.graph castTensor:nanIndex toType:indexOut.dataType name:nil];
        }
        indexOut = [ctx.graph selectWithPredicateTensor:anyNaN
                                    truePredicateTensor:nanIndex
                                   falsePredicateTensor:indexOut
                                                   name:nil];
        MPSGraphTensor* nanValue = [ctx.graph constantWithScalar:NAN dataType:valueOut.dataType];
        valueOut = [ctx.graph selectWithPredicateTensor:anyNaN
                                    truePredicateTensor:nanValue
                                   falsePredicateTensor:valueOut
                                                   name:nil];
    }

    MPSDataType valueType = GetResultMpsType(ctx.op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [ctx.graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(ctx.op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [ctx.graph castTensor:indexOut toType:indexType name:nil];
    }

    ProcessResult combined = CombineArgReduceInit(ctx, reduceOp, kind, valueOut, indexOut);
    if (!combined.ok()) {
        return combined;
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(ctx.op, 0);
    if (valueShape && valueOut) {
        valueOut = [ctx.graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(ctx.op, 1);
    if (indexShape && indexOut) {
        indexOut = [ctx.graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = valueOut;
    ctx.values[ctx.op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

static ProcessResult HandleReduce(HandlerContext& ctx) {
    if (ctx.op->getNumResults() > 1) {
        return HandleMultiResultReduce(ctx);
    }
    return HandleSingleResultReduce(ctx);
}
REGISTER_MPS_OP("stablehlo.reduce", HandleReduce);

// MPSGraph max pooling returns the largest non-NaN value, so a NaN inside a
// window was dropped where XLA propagates it; it keeps one element on a +-0
// tie, where XLA orders -0 < +0; and a window holding only -inf came back as 0
// (min pooling: -0 for only +inf), padded or not (build/probe_r20c_poolinf.py).
// Max-pool one priority indicator with the same descriptor: 3 NaN, 2 the
// winning zero (+0 for max, -0 for min), 1 any other value, 0 the losing
// infinity (-inf for max, +inf for min). A pooled 0 means the window held only
// the losing infinity. `source` is the un-negated pooling source (its signs are
// the real ones) and `pooled` the final max (or negated-back min) value in the
// pooled layout, so the fix-up does not depend on how MPSGraph negates +-0.
// The losing-inf fix rides on the same pass, so setting both opt-outs below
// re-exposes it.
static MPSGraphTensor* PoolingExtremaFixup(MPSGraph* g, MPSGraphTensor* source,
                                           MPSGraphPooling2DOpDescriptor* desc, MPSGraphTensor* pooled,
                                           bool isMax) {
    static const bool nanDisabled = []() {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_POOLING_NAN_PROPAGATION");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    MPSDataType st = source.dataType;
    const bool doTie = WindowZeroTieEnabled();
    if ((nanDisabled && !doTie) || !pooled ||
        (st != MPSDataTypeFloat32 && st != MPSDataTypeFloat16 && st != MPSDataTypeBFloat16)) {
        return pooled;
    }
    // Costs one extra pooling pass (~0.39 ms on the sliding_max benches). Gating
    // it behind an MPSGraph `if` on any(isNaN) was measured at ~3.7 ms per call,
    // far worse, so the pass is unconditional. The zero bit shares the pass.
    // JAX_METALLIB_DISABLE_POOLING_NAN_PROPAGATION=1 drops the NaN bit and
    // JAX_METALLIB_DISABLE_WINDOW_ZERO_TIE=1 the zero bit.
    auto f32 = [&](double v) { return [g constantWithScalar:v dataType:MPSDataTypeFloat32]; };
    const double losingInf = isMax ? -INFINITY : INFINITY;
    // NaN != -inf is true, so NaN starts at 1 and is raised to 3 below.
    MPSGraphTensor* flag = [g castTensor:[g notEqualWithPrimaryTensor:source
                                                      secondaryTensor:[g constantWithScalar:losingInf
                                                                                   dataType:st]
                                                                 name:nil]
                                  toType:MPSDataTypeFloat32
                                    name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:st];
    if (doTie) {
        MPSGraphTensor* isZero = [g equalWithPrimaryTensor:source secondaryTensor:zero name:nil];
        MPSGraphTensor* neg = [g signbitWithTensor:source name:nil];
        MPSGraphTensor* winSign = isMax ? [g notWithTensor:neg name:nil] : neg;
        flag = [g selectWithPredicateTensor:[g logicalANDWithPrimaryTensor:isZero
                                                           secondaryTensor:winSign
                                                                      name:nil]
                        truePredicateTensor:f32(2.0)
                       falsePredicateTensor:flag
                                       name:nil];
    }
    if (!nanDisabled) {
        flag = [g selectWithPredicateTensor:[g isNaNWithTensor:source name:nil]
                        truePredicateTensor:f32(3.0)
                       falsePredicateTensor:flag
                                       name:nil];
    }
    MPSGraphTensor* any = [g maxPooling2DWithSourceTensor:flag descriptor:desc name:nil];
    MPSGraphTensor* out = pooled;
    MPSGraphTensor* allLosing = [g lessThanWithPrimaryTensor:any secondaryTensor:f32(0.5) name:nil];
    out = [g selectWithPredicateTensor:allLosing
                   truePredicateTensor:[g constantWithScalar:losingInf dataType:pooled.dataType]
                  falsePredicateTensor:out
                                  name:nil];
    if (doTie) {
        MPSGraphTensor* hasWin = [g greaterThanOrEqualToWithPrimaryTensor:any
                                                          secondaryTensor:f32(2.0)
                                                                     name:nil];
        MPSGraphTensor* pz = [g equalWithPrimaryTensor:pooled
                                       secondaryTensor:[g constantWithScalar:0.0 dataType:pooled.dataType]
                                                  name:nil];
        MPSGraphTensor* winZero = [g constantWithScalar:(isMax ? 0.0 : -0.0) dataType:pooled.dataType];
        out = [g selectWithPredicateTensor:[g logicalANDWithPrimaryTensor:hasWin secondaryTensor:pz name:nil]
                       truePredicateTensor:winZero
                      falsePredicateTensor:out
                                      name:nil];
    }
    if (!nanDisabled) {
        MPSGraphTensor* hasNaN = [g greaterThanOrEqualToWithPrimaryTensor:any
                                                          secondaryTensor:f32(3.0)
                                                                     name:nil];
        MPSGraphTensor* nanT = [g constantWithScalar:NAN dataType:pooled.dataType];
        out = [g selectWithPredicateTensor:hasNaN truePredicateTensor:nanT falsePredicateTensor:out name:nil];
    }
    return out;
}

// Structural match for the combiner JAX emits for lax.cumlogsumexp:
//   d = a - b; select(d != d, a + b, max(a, b) + log1p(exp(-|d|)))
// (operand order of the commutative ops is not significant).
static bool IsLogAddExpRegion(mlir::Region& body) {
    if (body.empty()) {
        return false;
    }
    mlir::Block& block = body.front();
    if (block.getNumArguments() != 2 || block.empty()) {
        return false;
    }
    mlir::Value x = block.getArgument(0);
    mlir::Value y = block.getArgument(1);
    auto isArgPair = [&](mlir::Operation* op) {
        if (!op || op->getNumOperands() != 2) {
            return false;
        }
        mlir::Value p = op->getOperand(0);
        mlir::Value q = op->getOperand(1);
        return (p == x && q == y) || (p == y && q == x);
    };
    mlir::Operation* term = &block.back();
    if (!mlir::isa<mlir::stablehlo::ReturnOp>(term) || term->getNumOperands() != 1) {
        return false;
    }
    auto sel = term->getOperand(0).getDefiningOp<mlir::stablehlo::SelectOp>();
    if (!sel) {
        return false;
    }
    auto cmp = sel.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
    auto sum = sel.getOnTrue().getDefiningOp<mlir::stablehlo::AddOp>();
    auto res = sel.getOnFalse().getDefiningOp<mlir::stablehlo::AddOp>();
    if (!cmp || !sum || !res || cmp.getComparisonDirection() != mlir::stablehlo::ComparisonDirection::NE ||
        cmp.getLhs() != cmp.getRhs()) {
        return false;
    }
    auto diff = cmp.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>();
    if (!diff || !isArgPair(diff.getOperation()) || !isArgPair(sum.getOperation())) {
        return false;
    }
    auto mx = res.getLhs().getDefiningOp<mlir::stablehlo::MaxOp>();
    auto lp = res.getRhs().getDefiningOp<mlir::stablehlo::Log1pOp>();
    if (!mx || !lp) {
        mx = res.getRhs().getDefiningOp<mlir::stablehlo::MaxOp>();
        lp = res.getLhs().getDefiningOp<mlir::stablehlo::Log1pOp>();
    }
    if (!mx || !lp || !isArgPair(mx.getOperation())) {
        return false;
    }
    auto ex = lp.getOperand().getDefiningOp<mlir::stablehlo::ExpOp>();
    auto ng = ex ? ex.getOperand().getDefiningOp<mlir::stablehlo::NegOp>() : nullptr;
    auto ab = ng ? ng.getOperand().getDefiningOp<mlir::stablehlo::AbsOp>() : nullptr;
    return ab && ab.getOperand() == diff.getResult();
}

// logaddexp with the same NaN/inf rules as the JAX region. log1p(e) for
// e = exp(-|d|) in [0, 1] uses the u = 1 + e correction, exact when u == 1.
static MPSGraphTensor* LogAddExpTensors(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
    MPSGraphTensor* d = [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    MPSGraphTensor* isNan = [g isNaNWithTensor:d name:nil];
    MPSGraphTensor* sum = [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    MPSGraphTensor* mx = [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
    MPSGraphTensor* e =
        [g exponentWithTensor:[g negativeWithTensor:[g absoluteWithTensor:d name:nil] name:nil] name:nil];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:e.dataType];
    MPSGraphTensor* u = [g additionWithPrimaryTensor:one secondaryTensor:e name:nil];
    MPSGraphTensor* um1 = [g subtractionWithPrimaryTensor:u secondaryTensor:one name:nil];
    MPSGraphTensor* lp = [g multiplicationWithPrimaryTensor:[g logarithmWithTensor:u name:nil]
                                            secondaryTensor:[g divisionWithPrimaryTensor:e
                                                                         secondaryTensor:um1
                                                                                    name:nil]
                                                       name:nil];
    MPSGraphTensor* exact = [g equalWithPrimaryTensor:u secondaryTensor:one name:nil];
    lp = [g selectWithPredicateTensor:exact truePredicateTensor:e falsePredicateTensor:lp name:nil];
    MPSGraphTensor* r = [g additionWithPrimaryTensor:mx secondaryTensor:lp name:nil];
    return [g selectWithPredicateTensor:isNan truePredicateTensor:sum falsePredicateTensor:r name:nil];
}

// Full cumulative logaddexp window (lax.cumlogsumexp): window == axis length with
// (n-1, 0) or (0, n-1) padding, unit strides/dilations. Hillis-Steele scan in
// log2(n) steps, computed in f32. Window i (after the optional reverse) also
// holds n-1-i padding slots of init plus the init fold, i.e. n-i copies of
// init, and logaddexp of k copies of c is c + log(k). Returns nil otherwise.
static MPSGraphTensor* LowerCumulativeLogAddExp(HandlerContext& ctx, MPSGraphTensor* input,
                                                llvm::ArrayRef<int64_t> win, llvm::ArrayRef<int64_t> strides,
                                                llvm::ArrayRef<int64_t> baseDil,
                                                llvm::ArrayRef<int64_t> winDil,
                                                const std::vector<std::pair<int64_t, int64_t>>& padding) {
    MPSGraph* g = ctx.graph;
    MPSGraphTensor* init = GetInputTensor(ctx, 1);
    size_t rank = win.size();
    MPSDataType dt = input.dataType;
    bool isFloat = dt == MPSDataTypeFloat32 || dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
    if (!isFloat || !init || !input.shape || input.shape.count != rank || padding.size() != rank ||
        strides.size() != rank || baseDil.size() != rank || winDil.size() != rank) {
        return nil;
    }
    NSInteger axis = -1;
    bool reverse = false;
    for (size_t d = 0; d < rank; ++d) {
        int64_t n = [input.shape[d] longLongValue];
        if (strides[d] != 1 || baseDil[d] != 1 || winDil[d] != 1 || n <= 0) {
            return nil;
        }
        if (win[d] == 1 && padding[d].first == 0 && padding[d].second == 0) {
            continue;
        }
        if (axis != -1 || win[d] != n) {
            return nil;
        }
        if (padding[d].first == n - 1 && padding[d].second == 0) {
            reverse = false;
        } else if (padding[d].first == 0 && padding[d].second == n - 1) {
            reverse = true;
        } else {
            return nil;
        }
        axis = (NSInteger)d;
    }

    const MPSDataType work = MPSDataTypeFloat32;
    MPSGraphTensor* x = dt == work ? input : [g castTensor:input toType:work name:nil];
    MPSGraphTensor* initW = init.dataType == work ? init : [g castTensor:init toType:work name:nil];
    MPSGraphTensor* result = nil;
    if (axis < 0) {
        result = LogAddExpTensors(g, x, initW);
    } else {
        NSArray<NSNumber*>* axes = @[@(axis)];
        if (reverse) {
            x = [g reverseTensor:x axes:axes name:nil];
        }
        NSArray<NSNumber*>* shape = x.shape;
        int64_t n = [shape[(NSUInteger)axis] longLongValue];
        MPSGraphTensor* cur = x;
        for (int64_t s = 1; s < n; s *= 2) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* steps = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
            for (size_t d = 0; d < rank; ++d) {
                bool onAxis = (NSInteger)d == axis;
                [starts addObject:@0];
                [ends addObject:onAxis ? @(n - s) : shape[d]];
                [steps addObject:@1];
                [left addObject:onAxis ? @(s) : @0];
                [right addObject:@0];
            }
            MPSGraphTensor* head = [g sliceTensor:cur starts:starts ends:ends strides:steps name:nil];
            MPSGraphTensor* shifted = [g padTensor:head
                                   withPaddingMode:MPSGraphPaddingModeConstant
                                       leftPadding:left
                                      rightPadding:right
                                     constantValue:-INFINITY
                                              name:nil];
            cur = LogAddExpTensors(g, cur, shifted);
        }
        std::vector<float> logk((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            logk[(size_t)i] = (float)std::log((double)(n - i));
        }
        NSMutableArray<NSNumber*>* kshape = [NSMutableArray arrayWithCapacity:rank];
        for (size_t d = 0; d < rank; ++d) {
            [kshape addObject:(NSInteger)d == axis ? @(n) : @1];
        }
        MPSGraphTensor* lk = [g constantWithData:[NSData dataWithBytes:logk.data()
                                                                length:logk.size() * sizeof(float)]
                                           shape:kshape
                                        dataType:work];
        MPSGraphTensor* initTerm = [g additionWithPrimaryTensor:initW secondaryTensor:lk name:nil];
        result = LogAddExpTensors(g, cur, initTerm);
        if (reverse) {
            result = [g reverseTensor:result axes:axes name:nil];
        }
    }
    if (dt != work) {
        result = [g castTensor:result toType:dt name:nil];
    }
    return result;
}

// reducerOverride replaces the body's reducer name (the bool wrapper lowers
// or/and on the 0/1 widening as max/min).
// XLA:CPU float window max/min flushes f32 subnormals to a signed zero. Pooling,
// cumulative and slice-fold results outside the float order-key gate kept them
// (build/probe_r20e_subnormal_map.log: 445 f32 mismatches, all ref=+-0 got=+-sub).
// Flushing is monotone in the -0 < +0 order, so flushing the result equals
// flushing the operand. Integer form: exponent bits 0 -> keep only the sign bit,
// exact whatever MPSGraph's float compares do with subnormals.
// JAX_METALLIB_DISABLE_WINDOW_SUBNORMAL_FLUSH=1 skips it.
// JAX_METALLIB_WINDOW_SUBNORMAL_FLUSH_BITS=1 uses the integer-bit flush instead
// of the float select (its reinterpret casts kept the post-pass from stitching:
// f32 cummax 4096^2 +2.0 ms, build/bench_window_subnormal_flush_r20e.log).
static bool WindowSubnormalFlushBits() {
    static const bool bits = []() {
        const char* v = std::getenv("JAX_METALLIB_WINDOW_SUBNORMAL_FLUSH_BITS");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return bits;
}

static MPSGraphTensor* FlushF32SubnormalSigned(MPSGraph* g, MPSGraphTensor* t, bool forceBits) {
    if (!forceBits && !WindowSubnormalFlushBits()) {
        // |x| < 2^-126 holds for a subnormal whether or not the compare flushes
        // it, and x * 0 keeps the sign of every finite x.
        MPSGraphTensor* tiny = [g lessThanWithPrimaryTensor:[g absoluteWithTensor:t name:nil]
                                            secondaryTensor:[g constantWithScalar:1.1754943508222875e-38
                                                                         dataType:MPSDataTypeFloat32]
                                                       name:nil];
        MPSGraphTensor* zero = [g multiplicationWithPrimaryTensor:t
                                                  secondaryTensor:[g constantWithScalar:0.0
                                                                               dataType:MPSDataTypeFloat32]
                                                             name:nil];
        return [g selectWithPredicateTensor:tiny truePredicateTensor:zero falsePredicateTensor:t name:nil];
    }
    // reinterpret_cast rejects rank-0 operands (a runtime init aborted the
    // process: build/probe_r20e_runtime_init.log), so flush those as [1].
    if (t.shape && t.shape.count == 0) {
        MPSGraphTensor* flat =
            FlushF32SubnormalSigned(g, [g reshapeTensor:t withShape:@[@1] name:nil], forceBits);
        return [g reshapeTensor:flat withShape:@[] name:nil];
    }
    MPSGraphTensor* bits = [g reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* expo = [g bitwiseANDWithPrimaryTensor:bits
                                          secondaryTensor:OrderKeyConst(g, MPSDataTypeInt32, 0x7F800000)
                                                     name:nil];
    MPSGraphTensor* sub = [g equalWithPrimaryTensor:expo
                                    secondaryTensor:OrderKeyConst(g, MPSDataTypeInt32, 0)
                                               name:nil];
    MPSGraphTensor* sign = [g bitwiseANDWithPrimaryTensor:bits
                                          secondaryTensor:OrderKeyConst(g, MPSDataTypeInt32, INT32_MIN)
                                                     name:nil];
    bits = [g selectWithPredicateTensor:sub truePredicateTensor:sign falsePredicateTensor:bits name:nil];
    return [g reinterpretCastTensor:bits toType:MPSDataTypeFloat32 name:nil];
}

static ProcessResult HandleReduceWindowImpl(HandlerContext& ctx, const char* reducerOverride = nullptr,
                                            bool* skipSubnormalFlush = nullptr) {
    auto reduceWindowOp = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(ctx.op);
    if (!reduceWindowOp) {
        return ProcessResult::Error("reduce_window: expected ReduceWindowOp");
    }
    if (ctx.op->getNumOperands() != 2 || ctx.op->getNumResults() != 1) {
        return ProcessResult::Error(
            "reduce_window: only single-input single-result reduce_window is supported");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("reduce_window: input tensor not found");
    }

    auto windowDimensions = reduceWindowOp.getWindowDimensions();
    size_t rank = windowDimensions.size();
    if (rank == 0) {
        return ProcessResult::Error("reduce_window: rank-0 reduce_window is not supported");
    }

    std::vector<int64_t> windowStrides;
    std::vector<int64_t> baseDilations;
    std::vector<int64_t> windowDilations;
    std::vector<std::pair<int64_t, int64_t>> padding;
    if (!ReadArrayOrDefaultOnes(reduceWindowOp.getWindowStrides(), rank, windowStrides)) {
        return ProcessResult::Error("reduce_window: invalid window_strides rank mismatch");
    }
    if (!ReadArrayOrDefaultOnes(reduceWindowOp.getBaseDilations(), rank, baseDilations)) {
        return ProcessResult::Error("reduce_window: invalid base_dilations rank mismatch");
    }
    if (!ReadArrayOrDefaultOnes(reduceWindowOp.getWindowDilations(), rank, windowDilations)) {
        return ProcessResult::Error("reduce_window: invalid window_dilations rank mismatch");
    }
    if (!ReadPaddingOrDefaultZeros(reduceWindowOp.getPadding(), rank, padding)) {
        return ProcessResult::Error("reduce_window: invalid padding shape");
    }

    std::string reducerType =
        reducerOverride ? std::string(reducerOverride) : GetReductionOpType(reduceWindowOp.getBody());
    if (reducerType.rfind("unsupported_region(", 0) == 0 && IsLogAddExpRegion(reduceWindowOp.getBody())) {
        if (MPSGraphTensor* scanned = LowerCumulativeLogAddExp(ctx, input, windowDimensions, windowStrides,
                                                               baseDilations, windowDilations, padding)) {
            return Result(ctx, scanned, "reduce_window");
        }
        return ProcessResult::Error("reduce_window: a logaddexp reducer is only supported as a full "
                                    "cumulative scan (lax.cumlogsumexp)");
    }

    // A max/min window of all ones (stride 1, no padding or dilation) reduces
    // nothing: the result is reducer(x, init). Such windows (lax.cummax over a
    // length-1 axis) fell into the pooling route, which returned 0.0 for
    // max(-inf) and -0.0 for min(+inf) and rejected bf16 min at rank 3.
    if (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") {
        bool identityWindow = input.dataType != MPSDataTypeBool;
        for (size_t d = 0; identityWindow && d < rank; ++d) {
            identityWindow = windowDimensions[d] == 1 && windowStrides[d] == 1 && baseDilations[d] == 1 &&
                             windowDilations[d] == 1 && padding[d].first == 0 && padding[d].second == 0;
        }
        MPSGraphTensor* init = identityWindow ? GetInputTensor(ctx, 1) : nil;
        if (init) {
            if (init.dataType != input.dataType) {
                init = [ctx.graph castTensor:init toType:input.dataType name:nil];
            }
            MPSGraphTensor* result = WindowMaxMin(ctx.graph, input, init, reducerType == "stablehlo.maximum",
                                                  ctx.op->getOperand(1));
            NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
            if (outputShape && ![outputShape isEqualToArray:result.shape]) {
                result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
            }
            return Result(ctx, result, "reduce_window");
        }
    }

    // base_dilation > 1 was rejected before any lowering ran. Materialise the
    // dilated operand instead (per axis: reshape to [n,1], concat k-1 filler
    // slots, reshape to n*k, trim to (n-1)*k+1) and continue with dilation 1.
    // Holes hold the reducer identity for add/multiply and init for max/min,
    // matching XLA:CPU (holes do not contribute; init is folded once).
    bool hasBaseDilation = false;
    for (int64_t d : baseDilations) {
        hasBaseDilation = hasBaseDilation || d != 1;
    }
    if (hasBaseDilation) {
        const bool isAdd = reducerType == "stablehlo.add";
        const bool isMul = reducerType == "stablehlo.multiply";
        const bool isExtrema = reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum";
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        bool supported = (isAdd || isMul || isExtrema) && input.shape && input.shape.count == rank &&
                         input.dataType != MPSDataTypeBool && (!isExtrema || init);
        for (size_t d = 0; supported && d < rank; ++d) {
            supported = baseDilations[d] >= 1 && input.shape[d].longLongValue >= 1;
        }
        if (!supported) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error("reduce_window: base_dilations != 1 is unsupported (" +
                                        DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                    windowStrides, baseDilations,
                                                                    windowDilations, padding, reducerType) +
                                        ")");
#else
            return ProcessResult::Error("reduce_window: base_dilations != 1 is unsupported");
#endif
        }
        MPSGraphTensor* fill = nil;
        if (isExtrema) {
            fill = init.dataType == input.dataType
                       ? init
                       : [ctx.graph castTensor:init toType:input.dataType name:nil];
        } else {
            fill = [ctx.graph constantWithScalar:(isMul ? 1.0 : 0.0) dataType:input.dataType];
        }
        for (size_t d = 0; d < rank; ++d) {
            const int64_t k = baseDilations[d];
            if (k == 1) {
                continue;
            }
            NSArray<NSNumber*>* shape = input.shape;
            const int64_t n = shape[d].longLongValue;
            NSMutableArray<NSNumber*>* column = [shape mutableCopy];
            [column insertObject:@1 atIndex:d + 1];
            NSMutableArray<NSNumber*>* holes = [shape mutableCopy];
            [holes insertObject:@(k - 1) atIndex:d + 1];
            NSMutableArray<NSNumber*>* spread = [shape mutableCopy];
            spread[d] = @(n * k);
            MPSGraphTensor* col = [ctx.graph reshapeTensor:input withShape:column name:nil];
            MPSGraphTensor* gap = [ctx.graph broadcastTensor:fill toShape:holes name:nil];
            MPSGraphTensor* joined = [ctx.graph concatTensors:@[col, gap]
                                                    dimension:static_cast<NSInteger>(d + 1)
                                                         name:nil];
            MPSGraphTensor* flat = [ctx.graph reshapeTensor:joined withShape:spread name:nil];
            input = [ctx.graph sliceTensor:flat
                                 dimension:static_cast<NSUInteger>(d)
                                     start:0
                                    length:(n - 1) * k + 1
                                      name:nil];
        }
        std::fill(baseDilations.begin(), baseDilations.end(), 1);
    }

    // Shared extrema slice fold: pad `src` with `padValue` (StableHLO pads with
    // init), take one strided/dilated slice per window position and fold with
    // plain max/min. Callers fold the init afterwards. nil when it does not fit.
    auto extremaSliceFold = [&](MPSGraphTensor* src, MPSGraphTensor* padValue, bool isMax,
                                bool blockFold = false) -> MPSGraphTensor* {
        int64_t positions = 1;
        std::vector<int64_t> outDims(rank, 0);
        MPSGraphTensor* padded = src;
        for (size_t d = 0; d < rank; ++d) {
            if (windowDimensions[d] < 1 || windowStrides[d] < 1 || windowDilations[d] < 1 ||
                padding[d].first < 0 || padding[d].second < 0) {
                return nil;
            }
            positions *= windowDimensions[d];
            if (positions > 256) {
                return nil;
            }
            int64_t size = [src.shape[d] longLongValue] + padding[d].first + padding[d].second;
            int64_t effective = (windowDimensions[d] - 1) * windowDilations[d] + 1;
            if (size < effective) {
                return nil;
            }
            outDims[d] = (size - effective) / windowStrides[d] + 1;
            NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:3];
            auto fill = [&](int64_t count) {
                NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:padded.shape];
                shape[d] = @(count);
                return [ctx.graph broadcastTensor:padValue toShape:shape name:nil];
            };
            if (padding[d].first > 0) {
                [parts addObject:fill(padding[d].first)];
            }
            [parts addObject:padded];
            if (padding[d].second > 0) {
                [parts addObject:fill(padding[d].second)];
            }
            if (parts.count > 1) {
                padded = [ctx.graph concatTensors:parts dimension:(NSInteger)d name:nil];
            }
        }
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape) {
            if (expectedShape.count != rank) {
                return nil;
            }
            for (size_t d = 0; d < rank; ++d) {
                if ([expectedShape[d] longLongValue] != outDims[d]) {
                    return nil;
                }
            }
        }
        // Window parameters of the fold below; the block prefix rewrites them.
        std::vector<int64_t> wDims(windowDimensions.begin(), windowDimensions.end());
        std::vector<int64_t> wStrides(windowStrides.begin(), windowStrides.end());
        std::vector<int64_t> wDils(windowDilations.begin(), windowDilations.end());
        // Block prefix (blockFold callers): an undilated axis whose window is a
        // multiple of its stride (w = m*s, s > 1) is cropped to (o + m - 1)*s,
        // reshaped to [o + m - 1, s] and max/min-reduced over the s axis, which
        // leaves a stride-1 window of m over the block extrema. This replaces the
        // strided slices (uncoalesced on the last axis) with one contiguous
        // reduction; exact because integer/key max/min is associative. f32 key
        // emulation on [4096,4096] (build/probe_r20f_reshape_fold.py): 2x2 s2
        // 3.63 -> 2.37 ms, 1x4 s4 4.59 -> 2.30, 1x4 s2 7.3 -> 3.56, 4x1 s2 5.05
        // -> 3.27. JAX_METALLIB_DISABLE_WINDOW_BLOCK_FOLD=1 keeps the strided slices.
        static const bool blockFoldDisabled = [] {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_WINDOW_BLOCK_FOLD");
            return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
        }();
        if (blockFold && !blockFoldDisabled) {
            bool anyBlock = false;
            NSMutableArray<NSNumber*>* cropStarts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* cropEnds = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* cropSteps = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* split = [NSMutableArray arrayWithCapacity:2 * rank];
            NSMutableArray<NSNumber*>* blockAxes = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* merged = [NSMutableArray arrayWithCapacity:rank];
            bool cropNeeded = false;
            for (size_t d = 0; d < rank; ++d) {
                const int64_t n = [padded.shape[d] longLongValue];
                const int64_t s = wStrides[d];
                const int64_t w = wDims[d];
                [cropStarts addObject:@0];
                [cropSteps addObject:@1];
                if (s > 1 && wDils[d] == 1 && w % s == 0) {
                    const int64_t m = w / s;
                    const int64_t blocks = outDims[d] + m - 1;
                    cropNeeded = cropNeeded || blocks * s != n;
                    [cropEnds addObject:@(blocks * s)];
                    [split addObject:@(blocks)];
                    [split addObject:@(s)];
                    [blockAxes addObject:@(split.count - 1)];
                    [merged addObject:@(blocks)];
                    wDims[d] = m;
                    wStrides[d] = 1;
                    anyBlock = true;
                } else {
                    [cropEnds addObject:@(n)];
                    [split addObject:@(n)];
                    [merged addObject:@(n)];
                }
            }
            if (anyBlock) {
                MPSGraphTensor* t = padded;
                if (cropNeeded) {
                    t = [ctx.graph sliceTensor:t starts:cropStarts ends:cropEnds strides:cropSteps name:nil];
                }
                t = [ctx.graph reshapeTensor:t withShape:split name:nil];
                t = isMax ? [ctx.graph reductionMaximumWithTensor:t axes:blockAxes name:nil]
                          : [ctx.graph reductionMinimumWithTensor:t axes:blockAxes name:nil];
                padded = [ctx.graph reshapeTensor:t withShape:merged name:nil];
            }
            // Remainder axes: an undilated axis with w = m*s + r (0 < r < s, m >= 1)
            // is padded/cropped to (o + m)*s and reshaped to [o + m, s]; one reduction
            // gives the full-block extrema, a second the extrema of the first r
            // columns, and out[i] = fold(full[i .. i+m-1], part[i+m]). The pad tail
            // only reaches full[o + m - 1], which is never read. f32 key emulation on
            // [4096,4096] (build/probe_r20g_twooff.py, "partial" vs strided): 1x3 s2
            // 4.02-4.04 vs 5.99-6.12 ms, 3x1 s2 3.25 vs 4.26-4.29, 1x5 s2 4.01-4.03
            // vs 8.52-8.54, 1x5 s3 3.52-3.54 vs 6.93, 3x3 s2 4.15-4.16 vs 6.09-6.12.
            // JAX_METALLIB_DISABLE_WINDOW_REMAINDER_FOLD=1 keeps the strided slices.
            static const bool remainderFoldDisabled = [] {
                const char* v = std::getenv("JAX_METALLIB_DISABLE_WINDOW_REMAINDER_FOLD");
                return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
            }();
            // `count` elements of `t` along d from `start`, unit stride, any rank.
            auto sliceAxis = [&](MPSGraphTensor* t, size_t d, int64_t start,
                                 int64_t count) -> MPSGraphTensor* {
                const NSUInteger tr = t.shape.count;
                NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:tr];
                NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:tr];
                NSMutableArray<NSNumber*>* steps = [NSMutableArray arrayWithCapacity:tr];
                for (NSUInteger e = 0; e < tr; ++e) {
                    const bool on = e == d;
                    [starts addObject:@(on ? start : 0)];
                    [ends addObject:@(on ? start + count : [t.shape[e] longLongValue])];
                    [steps addObject:@1];
                }
                return [ctx.graph sliceTensor:t starts:starts ends:ends strides:steps name:nil];
            };
            bool anyRemainder = false;
            for (size_t d = 0; d < rank && !remainderFoldDisabled; ++d) {
                const int64_t s = wStrides[d];
                const int64_t w = wDims[d];
                if (s <= 1 || wDils[d] != 1 || w <= s || w % s == 0) {
                    continue;
                }
                const int64_t m = w / s;
                const int64_t r = w % s;
                const int64_t o = outDims[d];
                const int64_t nb = o + m;
                const int64_t n = [padded.shape[d] longLongValue];
                MPSGraphTensor* t = padded;
                if (nb * s > n) {
                    NSMutableArray<NSNumber*>* tailShape = [NSMutableArray arrayWithArray:t.shape];
                    tailShape[d] = @(nb * s - n);
                    MPSGraphTensor* tailFill = [ctx.graph broadcastTensor:padValue
                                                                  toShape:tailShape
                                                                     name:nil];
                    t = [ctx.graph concatTensors:@[t, tailFill] dimension:(NSInteger)d name:nil];
                } else if (nb * s < n) {
                    t = sliceAxis(t, d, 0, nb * s);
                }
                NSMutableArray<NSNumber*>* rsplit = [NSMutableArray arrayWithCapacity:rank + 1];
                NSMutableArray<NSNumber*>* rmerged = [NSMutableArray arrayWithCapacity:rank];
                for (size_t e = 0; e < rank; ++e) {
                    if (e == d) {
                        [rsplit addObject:@(nb)];
                        [rsplit addObject:@(s)];
                        [rmerged addObject:@(nb)];
                    } else {
                        [rsplit addObject:t.shape[e]];
                        [rmerged addObject:t.shape[e]];
                    }
                }
                t = [ctx.graph reshapeTensor:t withShape:rsplit name:nil];
                NSArray<NSNumber*>* inner = @[@(d + 1)];
                MPSGraphTensor* full = isMax ? [ctx.graph reductionMaximumWithTensor:t axes:inner name:nil]
                                             : [ctx.graph reductionMinimumWithTensor:t axes:inner name:nil];
                MPSGraphTensor* part = sliceAxis(t, d + 1, 0, r);
                if (r > 1) {
                    part = isMax ? [ctx.graph reductionMaximumWithTensor:part axes:inner name:nil]
                                 : [ctx.graph reductionMinimumWithTensor:part axes:inner name:nil];
                }
                full = [ctx.graph reshapeTensor:full withShape:rmerged name:nil];
                part = [ctx.graph reshapeTensor:part withShape:rmerged name:nil];
                MPSGraphTensor* acc = sliceAxis(full, d, 0, o);
                for (int64_t j = 1; j < m; ++j) {
                    acc = KeyMaxMin(ctx.graph, acc, sliceAxis(full, d, j, o), isMax);
                }
                padded = KeyMaxMin(ctx.graph, acc, sliceAxis(part, d, m, o), isMax);
                wDims[d] = 1;
                wStrides[d] = 1;
                anyRemainder = true;
            }
            if (anyBlock || anyRemainder) {
                positions = 1;
                for (size_t d = 0; d < rank; ++d) {
                    positions *= wDims[d];
                }
                if (positions == 1) {
                    // Non-overlapping window: the blocks are the output, unless an
                    // unblocked window-1 axis still has a stride (e.g. window (1,2),
                    // strides (2,2)); the single-position slice below applies it.
                    bool needSlice = false;
                    for (size_t d = 0; d < rank; ++d) {
                        needSlice =
                            needSlice || wStrides[d] != 1 || [padded.shape[d] longLongValue] != outDims[d];
                    }
                    if (!needSlice) {
                        return padded;
                    }
                }
            }
        }
        // Fold structure ([4096,4096] int32 emulation, build/probe_r20d_fold_shapes.py):
        // one slice per position is fastest for small unit-stride windows (4x4 s1
        // 1.8 ms vs 2.6 separable) but grows superlinearly (1x64 84.8 ms). Folding
        // one axis at a time is exact (integer/key max/min is associative) and wins
        // for larger or strided multi-axis windows (5x5 s1 0.34x, 3x3 s2 0.73x,
        // 8x8 s8 0.53x). A stride-1 undilated axis of >= 32 positions folds by
        // doubling (m_2k = max(m_k, m_k shifted by k), then one combine; max is
        // idempotent, so the overlap is exact): 1x64 0.11x, 48x1 0.17x, 1x32 0.38x
        // (1x24 was 1.25x, so shorter axes keep the slice fold).
        // JAX_METALLIB_DISABLE_WINDOW_SEPARABLE_FOLD=1 keeps one slice per position.
        static const bool separableDisabled = [] {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_WINDOW_SEPARABLE_FOLD");
            return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
        }();
        constexpr int64_t kDoublingMinWindow = 32;
        int windowAxes = 0;
        bool anyStride = false;
        for (size_t d = 0; d < rank; ++d) {
            windowAxes += wDims[d] > 1 ? 1 : 0;
            anyStride = anyStride || wStrides[d] > 1;
        }
        // 2x2 s2 (4 positions) stays on the slice fold: separable was 1.18x
        // (bf16) / 1.20x (f32) slower there (build/bench_window_tie_r20c_gate_r20d.log).
        if (!separableDisabled && windowAxes >= 1 &&
            (positions > 16 || (anyStride && windowAxes > 1 && positions > 4))) {
            // `count` elements of `t` along d from `start` with `stride`.
            auto axisSlice = [&](MPSGraphTensor* t, size_t d, int64_t start, int64_t count,
                                 int64_t stride) -> MPSGraphTensor* {
                NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
                NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
                NSMutableArray<NSNumber*>* steps = [NSMutableArray arrayWithCapacity:rank];
                for (size_t e = 0; e < rank; ++e) {
                    const bool on = e == d;
                    [starts addObject:@(on ? start : 0)];
                    [ends addObject:@(on ? start + (count - 1) * stride + 1 : [t.shape[e] longLongValue])];
                    [steps addObject:@(on ? stride : 1)];
                }
                return [ctx.graph sliceTensor:t starts:starts ends:ends strides:steps name:nil];
            };
            MPSGraphTensor* t = padded;
            for (size_t d = 0; d < rank; ++d) {
                const int64_t w = wDims[d];
                const int64_t s = wStrides[d];
                const int64_t dil = wDils[d];
                const int64_t o = outDims[d];
                const int64_t n = [t.shape[d] longLongValue];
                if (w == 1) {
                    if (s != 1 || o != n) {
                        t = axisSlice(t, d, 0, o, s);
                    }
                    continue;
                }
                if (s == 1 && dil == 1 && w >= kDoublingMinWindow) {
                    // m[i] = fold(t[i : i + k]); after the loop k is the largest
                    // power of two <= w.
                    MPSGraphTensor* m = t;
                    int64_t k = 1;
                    int64_t len = n;
                    while (2 * k <= w) {
                        len -= k;
                        m = KeyMaxMin(ctx.graph, axisSlice(m, d, 0, len, 1), axisSlice(m, d, k, len, 1),
                                      isMax);
                        k *= 2;
                    }
                    MPSGraphTensor* a = axisSlice(m, d, 0, o, 1);
                    t = w == k ? a : KeyMaxMin(ctx.graph, a, axisSlice(m, d, w - k, o, 1), isMax);
                    continue;
                }
                MPSGraphTensor* r = nil;
                for (int64_t j = 0; j < w; ++j) {
                    MPSGraphTensor* sl = axisSlice(t, d, j * dil, o, s);
                    r = r ? KeyMaxMin(ctx.graph, r, sl, isMax) : sl;
                }
                t = r;
            }
            return t;
        }
        MPSGraphTensor* result = nil;
        for (int64_t p = 0; p < positions; ++p) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            std::vector<int64_t> offset(rank, 0);
            int64_t rem = p;
            for (size_t d = rank; d-- > 0;) {
                offset[d] = rem % wDims[d];
                rem /= wDims[d];
            }
            for (size_t d = 0; d < rank; ++d) {
                int64_t start = offset[d] * wDils[d];
                [starts addObject:@(start)];
                [ends addObject:@(start + (outDims[d] - 1) * wStrides[d] + 1)];
                [strides addObject:@(wStrides[d])];
            }
            MPSGraphTensor* slice = [ctx.graph sliceTensor:padded
                                                    starts:starts
                                                      ends:ends
                                                   strides:strides
                                                      name:nil];
            if (!result) {
                result = slice;
            } else {
                result = KeyMaxMin(ctx.graph, result, slice, isMax);
            }
        }
        return result;
    };

    // Integer max/min windows: the pooling and sliding paths below are float-only
    // (they pad with +/-inf), so integer extrema raised "unsupported pattern".
    // Lower exactly instead: pad with the init value (StableHLO pads with init),
    // take one strided/dilated slice per window position, and fold with max/min.
    auto integerExtremaWindow = [&]() -> MPSGraphTensor* {
        MPSDataType dt = input.dataType;
        const bool isInt = dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8 || dt == MPSDataTypeInt16 ||
                           dt == MPSDataTypeUInt16 || dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32;
        const bool isMax = reducerType == "stablehlo.maximum";
        if (!isInt || (!isMax && reducerType != "stablehlo.minimum") || !input.shape ||
            input.shape.count != rank) {
            return nil;
        }
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return nil;
        }
        // A full prefix scan (lax.cummax/cummin) has positions == L, so this
        // slice-fold costs L full passes: 4M-element i32 L=64 23.4 ms, L=256
        // 113 ms (the 256 -> 257 cliff was this gate, not MPSGraph). The direct
        // cumulative lowering below is ~0.3-1.0 ms from L=24 on, but slower for
        // short rows (i32 L=2 4.9 vs 2.4 ms), so long scans skip the fold:
        // 32-bit from L >= 12, 8/16-bit from L >= 24 (build/bench_int_scan_threshold.py).
        // JAX_METALLIB_DISABLE_INT_SCAN_DIRECT=1 keeps every scan on the slice-fold.
        static const bool scanDirectDisabled = [] {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_INT_SCAN_DIRECT");
            return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
        }();
        const bool wide = dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32;
        const int64_t kScanDirectMin = wide ? 12 : 24;
        if (!scanDirectDisabled) {
            int64_t scanLen = 0;
            int scanAxes = 0;
            bool fullScan = true;
            for (size_t d = 0; fullScan && d < rank; ++d) {
                const int64_t n = [input.shape[d] longLongValue];
                if (windowStrides[d] != 1 || windowDilations[d] != 1) {
                    fullScan = false;
                } else if (windowDimensions[d] > 1) {
                    ++scanAxes;
                    scanLen = n;
                    fullScan =
                        windowDimensions[d] == n && ((padding[d].first == n - 1 && padding[d].second == 0) ||
                                                     (padding[d].first == 0 && padding[d].second == n - 1));
                } else {
                    fullScan = padding[d].first == 0 && padding[d].second == 0;
                }
            }
            if (fullScan && scanAxes == 1 && scanLen >= kScanDirectMin) {
                return nil;
            }
        }
        if (init.dataType != dt) {
            init = [ctx.graph castTensor:init toType:dt name:nil];
        }
        MPSGraphTensor* result = extremaSliceFold(input, init, isMax);
        if (!result) {
            return nil;
        }
        return isMax ? [ctx.graph maximumWithPrimaryTensor:result secondaryTensor:init name:nil]
                     : [ctx.graph minimumWithPrimaryTensor:result secondaryTensor:init name:nil];
    };
    if (MPSGraphTensor* extrema = integerExtremaWindow()) {
        return Result(ctx, extrema, "reduce_window");
    }

    // Float max/min windows of rank <= 2 (sliding, strided, pooled; not full
    // prefix scans, whose cumulative lowering is already tie-exact at ~1.0x):
    // fold the order key (FloatToOrderKey, f32 subnormals flushed like XLA:CPU)
    // with plain integer max/min and decode once. Exact -0 < +0, NaN and flush
    // at every step; a JAX emulation was faster than pooling + indicator pass
    // ([4096,4096] f16 slide_max3 10.5 -> 3.2 ms, 2x2 pool 4.4 -> 2.0 ms; f32
    // 11.1 -> 6.4 and 4.4 -> 3.7 ms) but slower for rank-4 NHWC pooling, so
    // rank > 2 keeps the pooling path (build/probe_r20c_keyfold_emul.py). Pads
    // use key(-inf)/key(+inf): INT_MIN/INT_MAX decode to NaN. The init is
    // folded afterwards with the init-aware WindowMaxMin.
    // JAX_METALLIB_DISABLE_FLOAT_WINDOW_KEY=1 keeps the previous lowerings;
    // JAX_METALLIB_FLOAT_WINDOW_KEY_MAX_POSITIONS caps the window size.
    auto floatKeyExtremaWindow = [&]() -> MPSGraphTensor* {
        static const bool disabled = [] {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_FLOAT_WINDOW_KEY");
            return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
        }();
        static const int64_t maxPositionsOverride = [] {
            const char* v = std::getenv("JAX_METALLIB_FLOAT_WINDOW_KEY_MAX_POSITIONS");
            if (v && v[0] != '\0') {
                const long long n = std::atoll(v);
                if (n > 0) {
                    return static_cast<int64_t>(n);
                }
            }
            return static_cast<int64_t>(0);
        }();
        const MPSDataType dt = input.dataType;
        const bool isMax = reducerType == "stablehlo.maximum";
        if (disabled || !WindowZeroTieEnabled() ||
            (dt != MPSDataTypeFloat32 && dt != MPSDataTypeFloat16 && dt != MPSDataTypeBFloat16) ||
            (!isMax && reducerType != "stablehlo.minimum") || rank == 0 || rank > 2 || !input.shape ||
            input.shape.count != rank) {
            return nil;
        }
        int64_t positions = 1;
        int scanAxes = 0;
        bool fullScan = true;
        for (size_t d = 0; d < rank; ++d) {
            const int64_t n = [input.shape[d] longLongValue];
            if (n <= 0) {
                return nil;
            }
            positions *= std::max<int64_t>(windowDimensions[d], 1);
            if (windowStrides[d] != 1 || windowDilations[d] != 1) {
                fullScan = false;
            } else if (windowDimensions[d] > 1) {
                ++scanAxes;
                fullScan = fullScan && windowDimensions[d] == n &&
                           ((padding[d].first == n - 1 && padding[d].second == 0) ||
                            (padding[d].first == 0 && padding[d].second == n - 1));
            } else {
                fullScan = fullScan && padding[d].first == 0 && padding[d].second == 0;
            }
        }
        // Measured crossovers ([4096,4096], build/bench_float_window_key_positions_r20c.py,
        // build/bench_float_window_key_shape_r20d.py, with the separable/doubling
        // fold): f16/bf16 win at every size tried (up to 64 positions); f32
        // stride-1 windows win up to 64 (5x5 0.32x, 1x64 0.21x, 64x1 0.16x) but
        // strided ones only up to 4 (3x3 s2 1.44x, 4x4 s4 1.06x, 8x8 s8 1.03x,
        // 1x8 s8 1.72x).
        bool unitStride = true;
        for (size_t d = 0; d < rank; ++d) {
            unitStride = unitStride && windowStrides[d] == 1;
        }
        // With the block and remainder folds (round 20g) the f32 strided cap of 4 is
        // stale: an unpadded, undilated strided window with at most one windowed axis,
        // or whose windowed axes are all pure blocks (w % s == 0), beats pooling up to
        // 64 positions. Three held-out sets ([4096,4096], build/bench_r20g_gate_heldout{,2,3}.log):
        // bare 0.18-0.76x, behind relu 0.22-0.96x (1x9 s2 7.55 -> 4.05 ms, 10x1 s3
        // 6.12 -> 2.85, 4x8 s4 3.54 -> 1.91, rank-1 5 s2 at 2^24 19.4 -> 3.4).
        // Padded windows stay on pooling: the pad costs ~2.7 ms on the key path (3x3 s2
        // SAME 6.87 vs 4.42 ms, 5x3 s2 padded 1.78x; one padTensor, a float pad and padding
        // the reduced blocks were all still slower, build/probe_r20g_pad_cause.log). Two
        // windowed axes with a remainder axis run 0.96-1.21x behind relu on 13 of 14
        // held-out shapes (producer fusion boundary; 5x5 s2 is 0.40x only because pooling
        // takes 13 ms there) and stay on pooling too. Every windowed axis must be strided and every
        // window-1 axis must have stride 1: a stride-1 windowed axis next to a strided
        // window-1 axis lost 1.75-2.04x (1x8 s(2,1) 10.62 vs 5.95 ms, 8x1 s(1,2) 10.14 vs
        // 5.45, 1x16 s(3,1) 12.83 vs 7.32; build/bench_r20g_native_gate.log).
        // JAX_METALLIB_DISABLE_F32_STRIDED_KEY_WIDEN=1 restores the cap of 4.
        static const bool stridedWidenDisabled = [] {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_F32_STRIDED_KEY_WIDEN");
            return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
        }();
        bool stridedFoldable = !unitStride && !stridedWidenDisabled;
        int windowedAxes = 0;
        bool allPureBlocks = true;
        for (size_t d = 0; d < rank && stridedFoldable; ++d) {
            stridedFoldable = windowDilations[d] == 1 && padding[d].first == 0 && padding[d].second == 0;
            if (windowDimensions[d] > 1) {
                ++windowedAxes;
                // w < s leaves the remainder fold with m = 0, so it falls back to strided
                // slices: last-axis 1x5 s8 1.22-1.49x, 1x5 s6 1.33-1.56x, 1x7 s8 1.57-1.87x
                // (build/bench_r20g_native_gate_wlts.log). 6x1 s8 (0.71x bare, 1.00x behind
                // relu) and rank-1 5 s8 (0.68-0.71x) won there but are excluded too until a
                // held-out set supports a finer rule.
                stridedFoldable =
                    stridedFoldable && windowStrides[d] > 1 && windowDimensions[d] >= windowStrides[d];
                allPureBlocks = allPureBlocks && windowDimensions[d] % windowStrides[d] == 0;
            } else {
                stridedFoldable = stridedFoldable && windowStrides[d] == 1;
            }
        }
        stridedFoldable = stridedFoldable && (windowedAxes <= 1 || allPureBlocks);
        const int64_t maxPositions = maxPositionsOverride > 0   ? maxPositionsOverride
                                     : dt != MPSDataTypeFloat32 ? 64
                                     : unitStride               ? 64
                                     : stridedFoldable          ? 64
                                                                : 4;
        if (positions < 2 || positions > maxPositions || (fullScan && scanAxes == 1)) {
            return nil;
        }
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return nil;
        }
        const bool wide = dt == MPSDataTypeFloat32;
        const MPSDataType it = wide ? MPSDataTypeInt32 : MPSDataTypeInt16;
        const int64_t expMask = wide ? 0x7F800000 : (dt == MPSDataTypeFloat16 ? 0x7C00 : 0x7F80);
        MPSGraphTensor* key = FloatToOrderKey(ctx.graph, input, isMax, /*flushF32Subnormal=*/true);
        MPSGraphTensor* pad = OrderKeyConst(ctx.graph, it, isMax ? -expMask - 1 : expMask);
        MPSGraphTensor* folded = extremaSliceFold(key, pad, isMax, /*blockFold=*/true);
        if (!folded) {
            return nil;
        }
        MPSGraphTensor* result = OrderKeyToFloat(ctx.graph, folded, dt, isMax);
        return WindowMaxMin(ctx.graph, result, init, isMax, ctx.op->getOperand(1));
    };
    if (MPSGraphTensor* extrema = floatKeyExtremaWindow()) {
        if (skipSubnormalFlush) {
            *skipSubnormalFlush = true; // FloatToOrderKey already flushed f32 subnormals.
        }
        return Result(ctx, extrema, "reduce_window");
    }

    // Pooling accumulates each window independently, avoiding float32 prefix
    // cancellation for sums and unbounded cumulative max/min for extrema.
    auto poolSmallRankWindow = [&](NSInteger axis, int64_t window, int64_t padLow,
                                   int64_t padHigh) -> MPSGraphTensor* {
        if (rank > 2 || !input.shape || input.shape.count != rank ||
            (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32)) {
            return nil;
        }
        NSArray<NSNumber*>* shape =
            rank == 1 ? @[@1, @1, input.shape[0], @1] : @[@1, input.shape[0], input.shape[1], @1];
        MPSGraphTensor* source = [ctx.graph reshapeTensor:input withShape:shape name:nil];
        bool height = rank == 2 && axis == 0;
        NSUInteger kWidth = (NSUInteger)(height ? 1 : window);
        NSUInteger kHeight = (NSUInteger)(height ? window : 1);
        NSUInteger left = (NSUInteger)(height ? 0 : padLow);
        NSUInteger right = (NSUInteger)(height ? 0 : padHigh);
        NSUInteger top = (NSUInteger)(height ? padLow : 0);
        NSUInteger bottom = (NSUInteger)(height ? padHigh : 0);
        if (input.dataType == MPSDataTypeFloat16 &&
            (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum")) {
            // Half-precision max pooling with right padding can shift windows
            // to the left on MPSGraph. Extrema are exact after widening.
            source = [ctx.graph castTensor:source toType:MPSDataTypeFloat32 name:nil];
        }
        MPSGraphPooling2DOpDescriptor* desc =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:kWidth
                                                        kernelHeight:kHeight
                                                           strideInX:1
                                                           strideInY:1
                                                     dilationRateInX:1
                                                     dilationRateInY:1
                                                         paddingLeft:left
                                                        paddingRight:right
                                                          paddingTop:top
                                                       paddingBottom:bottom
                                                        paddingStyle:MPSGraphPaddingStyleExplicit
                                                          dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
        if (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") {
            const bool isMax = reducerType == "stablehlo.maximum";
            MPSGraphTensor* signSource = source;
            if (!isMax) {
                source = [ctx.graph negativeWithTensor:source name:nil];
            }
            MPSGraphTensor* pooled = [ctx.graph maxPooling2DWithSourceTensor:source descriptor:desc name:nil];
            if (!isMax) {
                pooled = [ctx.graph negativeWithTensor:pooled name:nil];
            }
            pooled = PoolingExtremaFixup(ctx.graph, signSource, desc, pooled, isMax);
            if (input.dataType == MPSDataTypeFloat16) {
                pooled = [ctx.graph castTensor:pooled toType:input.dataType name:nil];
            }
            MPSGraphTensor* result = [ctx.graph reshapeTensor:pooled
                                                    withShape:GetOutputShape(ctx.op)
                                                         name:nil];
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return nil;
            }
            return WindowMaxMin(ctx.graph, result, init, isMax, ctx.op->getOperand(1));
        }
        desc.includeZeroPadToAverage = YES;
        MPSGraphTensor* avg = [ctx.graph avgPooling2DWithSourceTensor:source descriptor:desc name:nil];
        MPSGraphTensor* scale = [ctx.graph constantWithScalar:(double)window dataType:input.dataType];
        MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:avg
                                                            secondaryTensor:scale
                                                                       name:nil];
        result = [ctx.graph reshapeTensor:result withShape:GetOutputShape(ctx.op) name:nil];
        if (!IsExactAddIdentityInit(ctx.op->getOperand(1))) {
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return nil;
            }
            result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
        }
        return result;
    };
    NSInteger cumulativeAxis = -1;
    bool reverseCumulative = false;
    bool patternMatches = true;
    for (size_t dim = 0; dim < rank; ++dim) {
        if (windowStrides[dim] != 1 || baseDilations[dim] != 1 || windowDilations[dim] != 1) {
            patternMatches = false;
            break;
        }

        int64_t window = windowDimensions[dim];
        int64_t padLow = padding[dim].first;
        int64_t padHigh = padding[dim].second;
        if (window > 1) {
            bool forward = padLow == window - 1 && padHigh == 0;
            bool reverse = padLow == 0 && padHigh == window - 1;
            if (cumulativeAxis != -1 || (!forward && !reverse)) {
                patternMatches = false;
                break;
            }
            cumulativeAxis = (NSInteger)dim;
            reverseCumulative = reverse;
        } else if (window == 1) {
            if (padLow != 0 || padHigh != 0) {
                patternMatches = false;
                break;
            }
        } else {
            patternMatches = false;
            break;
        }
    }

    // Sliding sums and extrema are not full-prefix reductions. Reuse the
    // NHWC/NWC pooling route for supported high-rank spatial axes.
    if (patternMatches && cumulativeAxis >= 0 &&
        (reducerType == "stablehlo.add" || reducerType == "stablehlo.maximum" ||
         reducerType == "stablehlo.minimum") &&
        rank >= 3 && input.shape && input.shape.count == rank &&
        ((rank == 3 && cumulativeAxis == 1) || (rank == 4 && (cumulativeAxis == 1 || cumulativeAxis == 2))) &&
        windowDimensions[(size_t)cumulativeAxis] < [input.shape[(NSUInteger)cumulativeAxis] longLongValue]) {
        patternMatches = false;
    }

    if (cumulativeAxis < 0 || !patternMatches) {
        // The transpose of a partial-window sum uses an unpadded VALID
        // reduce_window. Pool single-axis rank-1/2 sums directly; other
        // pool patterns are handled below.
        NSInteger validSumAxis = -1;
        bool validSum = rank <= 2 && reducerType == "stablehlo.add" && input.shape.count == rank &&
                        IsScalarZero(ctx.op->getOperand(1));
        for (size_t dim = 0; validSum && dim < rank; ++dim) {
            int64_t window = windowDimensions[dim];
            int64_t axisSize = [input.shape[dim] longLongValue];
            if (window < 1 || window > axisSize || windowStrides[dim] != 1 || baseDilations[dim] != 1 ||
                windowDilations[dim] != 1 || padding[dim].first != 0 || padding[dim].second != 0) {
                validSum = false;
            } else if (window > 1) {
                if (validSumAxis >= 0) {
                    validSum = false;
                } else {
                    validSumAxis = (NSInteger)dim;
                }
            }
        }
        if (validSum && validSumAxis >= 0) {
            int64_t window = windowDimensions[(size_t)validSumAxis];
            if (MPSGraphTensor* pooled = poolSmallRankWindow(validSumAxis, window, 0, 0)) {
                return Result(ctx, pooled, "reduce_window");
            }
            int64_t axisSize = [input.shape[(NSUInteger)validSumAxis] longLongValue];
            MPSGraphTensor* prefix = [ctx.graph cumulativeSumWithTensor:input axis:validSumAxis name:nil];
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* headEnds = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
            for (size_t dim = 0; dim < rank; ++dim) {
                bool axis = dim == (size_t)validSumAxis;
                [starts addObject:axis ? @(window - 1) : @0];
                [ends addObject:input.shape[dim]];
                [headEnds addObject:axis ? @(axisSize - window) : input.shape[dim]];
                [strides addObject:@1];
                [left addObject:axis ? @1 : @0];
                [right addObject:@0];
            }
            MPSGraphTensor* result = [ctx.graph sliceTensor:prefix
                                                     starts:starts
                                                       ends:ends
                                                    strides:strides
                                                       name:nil];
            if (axisSize > window) {
                for (size_t dim = 0; dim < rank; ++dim) {
                    starts[dim] = @0;
                }
                MPSGraphTensor* head = [ctx.graph sliceTensor:prefix
                                                       starts:starts
                                                         ends:headEnds
                                                      strides:strides
                                                         name:nil];
                MPSGraphTensor* shifted = [ctx.graph padTensor:head
                                               withPaddingMode:MPSGraphPaddingModeConstant
                                                   leftPadding:left
                                                  rightPadding:right
                                                 constantValue:0.0
                                                          name:nil];
                result = [ctx.graph subtractionWithPrimaryTensor:result secondaryTensor:shifted name:nil];
            }
            return Result(ctx, result, "reduce_window");
        }

        auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
        bool isSpatial2DPooling =
            (rank == 2 && inputType && inputType.getRank() == 2 && input.shape && input.shape.count == 2 &&
             (input.dataType == MPSDataTypeFloat16 || input.dataType == MPSDataTypeFloat32) &&
             (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") &&
             windowDimensions[0] >= 1 && windowDimensions[1] >= 1 && windowStrides[0] >= 1 &&
             windowStrides[1] >= 1 && baseDilations[0] == 1 && baseDilations[1] == 1 &&
             windowDilations[0] == 1 && windowDilations[1] == 1 && padding[0].first >= 0 &&
             padding[0].second >= 0 && padding[1].first >= 0 && padding[1].second >= 0);
        bool is2DPooling =
            (rank == 4 && inputType && inputType.getRank() == 4 && windowDimensions.size() == 4 &&
             windowStrides.size() == 4 && baseDilations.size() == 4 && windowDilations.size() == 4 &&
             windowDimensions[0] == 1 && windowDimensions[3] == 1 && windowStrides[0] == 1 &&
             windowStrides[3] == 1 && baseDilations[0] == 1 && baseDilations[1] == 1 &&
             baseDilations[2] == 1 && baseDilations[3] == 1 && windowDilations[0] == 1 &&
             windowDilations[3] == 1 && windowDilations[1] >= 1 && windowDilations[2] >= 1 &&
             padding[0].first == 0 && padding[0].second == 0 && padding[3].first == 0 &&
             padding[3].second == 0 && padding[1].first >= 0 && padding[1].second >= 0 &&
             padding[2].first >= 0 && padding[2].second >= 0 && windowDimensions[1] >= 1 &&
             windowDimensions[2] >= 1 && windowStrides[1] >= 1 && windowStrides[2] >= 1);

        bool is1DPooling =
            (rank == 3 && inputType && inputType.getRank() == 3 && windowDimensions.size() == 3 &&
             windowStrides.size() == 3 && baseDilations.size() == 3 && windowDilations.size() == 3 &&
             windowDimensions[0] == 1 && windowDimensions[2] == 1 && windowStrides[0] == 1 &&
             windowStrides[2] == 1 && baseDilations[0] == 1 && baseDilations[1] == 1 &&
             baseDilations[2] == 1 && windowDilations[0] == 1 && windowDilations[2] == 1 &&
             windowDilations[1] >= 1 && padding[0].first == 0 && padding[0].second == 0 &&
             padding[2].first == 0 && padding[2].second == 0 && padding[1].first >= 0 &&
             padding[1].second >= 0 && windowDimensions[1] >= 1 && windowStrides[1] >= 1);

        // bf16 sum pooling was rejected ("sum-pooling only supported for
        // float16/float32"); leave it to generalWindowFold below, which
        // accumulates bf16 add in f32 and folds init once, like XLA:CPU.
        const bool bf16SumPool = reducerType == "stablehlo.add" && input.dataType == MPSDataTypeBFloat16;
        if ((is2DPooling || is1DPooling || isSpatial2DPooling) && !bf16SumPool) {
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return ProcessResult::Error("reduce_window: missing pooling init tensor");
            }
            int64_t kH = is2DPooling ? windowDimensions[1] : isSpatial2DPooling ? windowDimensions[0] : 1;
            int64_t kW = is2DPooling ? windowDimensions[2] : windowDimensions[1];
            int64_t sH = is2DPooling ? windowStrides[1] : isSpatial2DPooling ? windowStrides[0] : 1;
            int64_t sW = is2DPooling ? windowStrides[2] : windowStrides[1];
            int64_t dH = is2DPooling ? windowDilations[1] : 1;
            int64_t dW = is2DPooling ? windowDilations[2] : windowDilations[1];
            int64_t padTop = is2DPooling ? padding[1].first : isSpatial2DPooling ? padding[0].first : 0;
            int64_t padBottom = is2DPooling ? padding[1].second : isSpatial2DPooling ? padding[0].second : 0;
            int64_t padLeft = is2DPooling ? padding[2].first : padding[1].first;
            int64_t padRight = is2DPooling ? padding[2].second : padding[1].second;

            if (isSpatial2DPooling) {
                input = [ctx.graph reshapeTensor:input
                                       withShape:@[@1, input.shape[0], input.shape[1], @1]
                                            name:nil];
            }
            if (is1DPooling) {
                if (!input.shape || input.shape.count != 3) {
                    return ProcessResult::Error("reduce_window: expected rank-3 NWC tensor for 1D pooling");
                }
                input = [ctx.graph reshapeTensor:input
                                       withShape:@[input.shape[0], @1, input.shape[1], input.shape[2]]
                                            name:nil];
            }

            MPSGraphPooling2DOpDescriptor* desc =
                [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kW
                                                            kernelHeight:(NSUInteger)kH
                                                               strideInX:(NSUInteger)sW
                                                               strideInY:(NSUInteger)sH
                                                         dilationRateInX:(NSUInteger)dW
                                                         dilationRateInY:(NSUInteger)dH
                                                             paddingLeft:(NSUInteger)padLeft
                                                            paddingRight:(NSUInteger)padRight
                                                              paddingTop:(NSUInteger)padTop
                                                           paddingBottom:(NSUInteger)padBottom
                                                            paddingStyle:MPSGraphPaddingStyleExplicit
                                                              dataLayout:MPSGraphTensorNamedDataLayoutNHWC];

            if (reducerType == "stablehlo.add") {

                if (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32) {
                    return ProcessResult::Error(
                        "reduce_window: sum-pooling only supported for float16/float32 (" +
                        DescribeReduceWindowPattern(reduceWindowOp, windowDimensions, windowStrides,
                                                    baseDilations, windowDilations, padding, reducerType) +
                        ")");
                }
                desc.includeZeroPadToAverage = YES;
                MPSGraphTensor* avg = [ctx.graph avgPooling2DWithSourceTensor:input descriptor:desc name:nil];
                MPSGraphTensor* scale = [ctx.graph constantWithScalar:(double)(kH * kW)
                                                             dataType:input.dataType];
                MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:avg
                                                                    secondaryTensor:scale
                                                                               name:nil];
                if (!IsExactAddIdentityInit(ctx.op->getOperand(1))) {
                    result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
                }
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }

            if (isSpatial2DPooling && input.dataType == MPSDataTypeFloat16) {
                // Right-padded float16 max pooling can select shifted windows.
                input = [ctx.graph castTensor:input toType:MPSDataTypeFloat32 name:nil];
            }
            if (reducerType == "stablehlo.maximum") {
                MPSGraphTensor* result = [ctx.graph maxPooling2DWithSourceTensor:input
                                                                      descriptor:desc
                                                                            name:nil];
                result = PoolingExtremaFixup(ctx.graph, input, desc, result, true);
                if (isSpatial2DPooling && result.dataType != init.dataType) {
                    result = [ctx.graph castTensor:result toType:init.dataType name:nil];
                }
                result = WindowMaxMin(ctx.graph, result, init, true, ctx.op->getOperand(1));
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }

            if (reducerType == "stablehlo.minimum") {

                // bf16 min pooling was rejected while bf16 max pooling ran on the
                // same path; negation is exact in bf16, so it is accepted too.
                if (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32 &&
                    input.dataType != MPSDataTypeBFloat16) {
                    return ProcessResult::Error(
                        "reduce_window: min-pooling only supported for float16/bfloat16/float32 (" +
                        DescribeReduceWindowPattern(reduceWindowOp, windowDimensions, windowStrides,
                                                    baseDilations, windowDilations, padding, reducerType) +
                        ")");
                }
                MPSGraphTensor* neg = [ctx.graph negativeWithTensor:input name:nil];
                MPSGraphTensor* pooled = [ctx.graph maxPooling2DWithSourceTensor:neg
                                                                      descriptor:desc
                                                                            name:nil];
                MPSGraphTensor* result = [ctx.graph negativeWithTensor:pooled name:nil];
                result = PoolingExtremaFixup(ctx.graph, input, desc, result, false);
                if (isSpatial2DPooling && result.dataType != init.dataType) {
                    result = [ctx.graph castTensor:result toType:init.dataType name:nil];
                }
                result = WindowMaxMin(ctx.graph, result, init, false, ctx.op->getOperand(1));
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }
        }

        // General fallback for strided / dilated / multi-axis windows that no
        // path above handles (strided moving sums and products at any rank,
        // rank-1 strided max/min were rejected as "unsupported pattern").
        // StableHLO semantics exactly: pad with the init value, take one
        // strided/dilated slice per window position, fold with the reducer,
        // then fold in init once. 8-bit add/multiply widen to int32 (MPSGraph
        // saturates 8-bit arithmetic); f16/bf16 add/multiply accumulate in
        // f32; float max/min propagate NaN. Bounded to 256 window positions.
        auto generalWindowFold = [&]() -> MPSGraphTensor* {
            const bool isAdd = reducerType == "stablehlo.add";
            const bool isMul = reducerType == "stablehlo.multiply";
            const bool isMax = reducerType == "stablehlo.maximum";
            const bool isMin = reducerType == "stablehlo.minimum";
            if (!(isAdd || isMul || isMax || isMin) || !input.shape || input.shape.count != rank) {
                return nil;
            }
            const MPSDataType dt = input.dataType;
            const bool isHalf = dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
            const bool isFloat = dt == MPSDataTypeFloat32 || isHalf;
            const bool isNarrowInt = dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8;
            const bool isInt = isNarrowInt || dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16 ||
                               dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32;
            if (!isFloat && !isInt) {
                return nil;
            }
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return nil;
            }
            if (init.dataType != dt) {
                init = [ctx.graph castTensor:init toType:dt name:nil];
            }
            MPSDataType work = dt;
            if ((isAdd || isMul) && isHalf) {
                work = MPSDataTypeFloat32;
            } else if ((isAdd || isMul) && isNarrowInt) {
                work = MPSDataTypeInt32;
            }
            MPSGraphTensor* src = work == dt ? input : [ctx.graph castTensor:input toType:work name:nil];
            MPSGraphTensor* initW = work == dt ? init : [ctx.graph castTensor:init toType:work name:nil];

            // XLA:CPU skips padded positions (identity), then folds init once;
            // for max/min padding with init is equivalent after that fold.
            MPSGraphTensor* padValue =
                (isAdd || isMul) ? [ctx.graph constantWithScalar:(isAdd ? 0.0 : 1.0) dataType:work] : initW;
            int64_t positions = 1;
            std::vector<int64_t> outDims(rank, 0);
            MPSGraphTensor* padded = src;
            for (size_t d = 0; d < rank; ++d) {
                if (windowDimensions[d] < 1 || windowStrides[d] < 1 || windowDilations[d] < 1 ||
                    padding[d].first < 0 || padding[d].second < 0) {
                    return nil;
                }
                positions *= windowDimensions[d];
                if (positions > 256) {
                    return nil;
                }
                int64_t size = [src.shape[d] longLongValue] + padding[d].first + padding[d].second;
                int64_t effective = (windowDimensions[d] - 1) * windowDilations[d] + 1;
                if (size < effective) {
                    return nil;
                }
                outDims[d] = (size - effective) / windowStrides[d] + 1;
                NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:3];
                auto fill = [&](int64_t count) {
                    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:padded.shape];
                    shape[d] = @(count);
                    return [ctx.graph broadcastTensor:padValue toShape:shape name:nil];
                };
                if (padding[d].first > 0) {
                    [parts addObject:fill(padding[d].first)];
                }
                [parts addObject:padded];
                if (padding[d].second > 0) {
                    [parts addObject:fill(padding[d].second)];
                }
                if (parts.count > 1) {
                    padded = [ctx.graph concatTensors:parts dimension:(NSInteger)d name:nil];
                }
            }
            NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
            if (expectedShape) {
                if (expectedShape.count != rank) {
                    return nil;
                }
                for (size_t d = 0; d < rank; ++d) {
                    if ([expectedShape[d] longLongValue] != outDims[d]) {
                        return nil;
                    }
                }
            }
            const bool keyFold = (isMax || isMin) && UseWindowKeyFold(padded);
            if (keyFold) {
                padded = FloatToOrderKey(ctx.graph, padded, isMax, /*flushF32Subnormal=*/false);
            }
            auto combine = [&](MPSGraphTensor* a, MPSGraphTensor* b) -> MPSGraphTensor* {
                if (isAdd) {
                    return [ctx.graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
                }
                if (isMul) {
                    return [ctx.graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
                }
                return keyFold ? KeyMaxMin(ctx.graph, a, b, isMax) : WindowMaxMin(ctx.graph, a, b, isMax);
            };
            MPSGraphTensor* result = nil;
            for (int64_t p = 0; p < positions; ++p) {
                NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
                NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
                NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
                std::vector<int64_t> offset(rank, 0);
                int64_t rem = p;
                for (size_t d = rank; d-- > 0;) {
                    offset[d] = rem % windowDimensions[d];
                    rem /= windowDimensions[d];
                }
                for (size_t d = 0; d < rank; ++d) {
                    int64_t start = offset[d] * windowDilations[d];
                    [starts addObject:@(start)];
                    [ends addObject:@(start + (outDims[d] - 1) * windowStrides[d] + 1)];
                    [strides addObject:@(windowStrides[d])];
                }
                MPSGraphTensor* slice = [ctx.graph sliceTensor:padded
                                                        starts:starts
                                                          ends:ends
                                                       strides:strides
                                                          name:nil];
                result = result ? combine(result, slice) : slice;
            }
            if (!result) {
                return nil;
            }
            if (keyFold) {
                result = OrderKeyToFloat(ctx.graph, result, dt, isMax);
            }
            // The init may be a +-0 constant: fold it with the init-aware rule.
            result = (isMax || isMin) ? WindowMaxMin(ctx.graph, result, initW, isMax, ctx.op->getOperand(1))
                                      : combine(result, initW);
            if (work != dt) {
                result = [ctx.graph castTensor:result toType:dt name:nil];
            }
            return result;
        };
        if (MPSGraphTensor* folded = generalWindowFold()) {
            return Result(ctx, folded, "reduce_window");
        }

        return ProcessResult::Error("reduce_window: unsupported pattern (" +
                                    DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                windowStrides, baseDilations, windowDilations,
                                                                padding, reducerType) +
                                    ")");
    }

    int64_t window = windowDimensions[(size_t)cumulativeAxis];
    int64_t axisSize = [input.shape[(NSUInteger)cumulativeAxis] longLongValue];
    if (window < axisSize && (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum")) {
        if (MPSGraphTensor* pooled =
                poolSmallRankWindow(cumulativeAxis, window, padding[(size_t)cumulativeAxis].first,
                                    padding[(size_t)cumulativeAxis].second)) {
            return Result(ctx, pooled, "reduce_window");
        }
    }
    if (window < axisSize && (reducerType == "stablehlo.multiply" || reducerType == "stablehlo.maximum" ||
                              reducerType == "stablehlo.minimum")) {
        // A bounded window is not a cumulative reduction. Materialize the small
        // windows directly; max/min cannot be undone from a cumulative prefix,
        // and dividing cumulative products is incorrect when an input is zero.
        if (window > 32 || (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32 &&
                            input.dataType != MPSDataTypeBFloat16)) {
            return ProcessResult::Error(
                "reduce_window: sliding multiply/max/min requires a float window <= 32");
        }
        double padValue = reducerType == "stablehlo.multiply"  ? 1.0
                          : reducerType == "stablehlo.maximum" ? -INFINITY
                                                               : INFINITY;
        NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
        for (size_t dim = 0; dim < rank; ++dim) {
            [left addObject:dim == (size_t)cumulativeAxis ? @(padding[dim].first) : @0];
            [right addObject:dim == (size_t)cumulativeAxis ? @(padding[dim].second) : @0];
        }
        MPSGraphTensor* padded = [ctx.graph padTensor:input
                                      withPaddingMode:MPSGraphPaddingModeConstant
                                          leftPadding:left
                                         rightPadding:right
                                        constantValue:padValue
                                                 name:nil];
        const bool slideIsMax = reducerType == "stablehlo.maximum";
        const bool slideKeyFold = reducerType != "stablehlo.multiply" && UseWindowKeyFold(padded);
        if (slideKeyFold) {
            padded = FloatToOrderKey(ctx.graph, padded, slideIsMax, /*flushF32Subnormal=*/false);
        }
        MPSGraphTensor* result = nil;
        for (int64_t offset = 0; offset < window; ++offset) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            for (size_t dim = 0; dim < rank; ++dim) {
                [starts addObject:dim == (size_t)cumulativeAxis ? @(offset) : @0];
                [ends addObject:dim == (size_t)cumulativeAxis ? @(offset + axisSize) : input.shape[dim]];
                [strides addObject:@1];
            }
            MPSGraphTensor* next = [ctx.graph sliceTensor:padded
                                                   starts:starts
                                                     ends:ends
                                                  strides:strides
                                                     name:nil];
            if (!result) {
                result = next;
            } else if (reducerType == "stablehlo.multiply") {
                result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:next name:nil];
            } else if (slideKeyFold) {
                result = KeyMaxMin(ctx.graph, result, next, slideIsMax);
            } else {
                result = WindowMaxMin(ctx.graph, result, next, reducerType == "stablehlo.maximum");
            }
        }
        if (slideKeyFold) {
            result = OrderKeyToFloat(ctx.graph, result, input.dataType, slideIsMax);
        }
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return ProcessResult::Error("reduce_window: missing sliding init tensor");
        }
        if (reducerType == "stablehlo.multiply") {
            result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else {
            result = WindowMaxMin(ctx.graph, result, init, reducerType == "stablehlo.maximum",
                                  ctx.op->getOperand(1));
        }
        return Result(ctx, result, "reduce_window");
    }
    if (reducerType == "stablehlo.add") {
        int64_t axisSize = [input.shape[(NSUInteger)cumulativeAxis] longLongValue];
        if (window < axisSize) {
            if (MPSGraphTensor* pooled =
                    poolSmallRankWindow(cumulativeAxis, window, padding[(size_t)cumulativeAxis].first,
                                     padding[(size_t)cumulativeAxis].second)) {
                return Result(ctx, pooled, "reduce_window");
            }
        }
    }
    // Short float max/min prefix scans: MPSGraph cumulative max/min plus the NaN
    // flag scan below cost 9.8 ms (f32) / 11.2 ms (f16) for 4M elements at L=2,
    // against ~1.6 ms at L >= 64. Folding the L columns with NaN-propagating
    // max/min needs no flag scan and no reverse passes; it was faster for f32
    // L <= 4 and f16/bf16 L <= 8 (build/bench_f32_short_scan.py).
    // JAX_METALLIB_DISABLE_SHORT_FLOAT_SCAN_FOLD=1 keeps the cumulative lowering.
    auto shortFloatScanFold = [&]() -> MPSGraphTensor* {
        static const bool disabled = [] {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_SHORT_FLOAT_SCAN_FOLD");
            return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
        }();
        const bool isMax = reducerType == "stablehlo.maximum";
        if (disabled || !(isMax || reducerType == "stablehlo.minimum") || !input.shape ||
            input.shape.count != rank) {
            return nil;
        }
        const MPSDataType dt = input.dataType;
        const bool half = dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
        if (!half && dt != MPSDataTypeFloat32) {
            return nil;
        }
        const int64_t n = axisSize;
        if (n < 2 || n > (half ? 8 : 4) || window < n) {
            return nil;
        }
        const bool keyFold = UseWindowKeyFold(input);
        MPSGraphTensor* src =
            keyFold ? FloatToOrderKey(ctx.graph, input, isMax, /*flushF32Subnormal=*/false) : input;
        std::vector<MPSGraphTensor*> running(static_cast<size_t>(n), nil);
        MPSGraphTensor* acc = nil;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t i = reverseCumulative ? n - 1 - k : k;
            MPSGraphTensor* col = [ctx.graph sliceTensor:src
                                               dimension:(NSUInteger)cumulativeAxis
                                                   start:i
                                                  length:1
                                                    name:nil];
            acc = !acc      ? col
                  : keyFold ? KeyMaxMin(ctx.graph, acc, col, isMax)
                            : WindowMaxMin(ctx.graph, acc, col, isMax);
            running[static_cast<size_t>(i)] = acc;
        }
        NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
        for (MPSGraphTensor* t : running) {
            [parts addObject:t];
        }
        MPSGraphTensor* cat = [ctx.graph concatTensors:parts dimension:cumulativeAxis name:nil];
        return keyFold ? OrderKeyToFloat(ctx.graph, cat, dt, isMax) : cat;
    };
    MPSGraphTensor* shortFold = shortFloatScanFold();
    NSArray<NSNumber*>* axes = @[@(cumulativeAxis)];
    if (reverseCumulative && !shortFold) {
        input = [ctx.graph reverseTensor:input axes:axes name:nil];
    }
    MPSGraphTensor* result = nullptr;
    if (shortFold) {
        result = shortFold;
    } else if (reducerType == "stablehlo.add") {
        result = [ctx.graph cumulativeSumWithTensor:input axis:cumulativeAxis name:nil];
    } else if (reducerType == "stablehlo.multiply") {
        result = [ctx.graph cumulativeProductWithTensor:input axis:cumulativeAxis name:nil];
    } else if (reducerType == "stablehlo.maximum") {
        result = [ctx.graph cumulativeMaximumWithTensor:input axis:cumulativeAxis name:nil];
    } else if (reducerType == "stablehlo.minimum") {
        result = [ctx.graph cumulativeMinimumWithTensor:input axis:cumulativeAxis name:nil];
    } else {
        result = nil;
    }
    // MPSGraph cumulative max/min skip NaN (lax.cummax([3, nan, 5]) gave
    // [3, 3, 5]); XLA propagates NaN from its position onward. They also
    // saturate infinities: a prefix of only -inf gave cummax -FLT_MAX (and
    // +inf gave cummin +FLT_MAX). Scan one int32 state over the same (possibly
    // reversed) input: for max -inf -> 0, finite -> 1, +inf -> 2, NaN -> 3 (min
    // swaps the infinities). Its running max says which special value, if any,
    // the exact prefix result is. Keep the flag scan a cummax: an int32 cumsum
    // is faster alone (1.39 vs 2.47 ms at [4096,4096]) but the whole f32 cummax
    // got slower with it (6.2 ms vs 5.0 ms, 3 runs), so do not swap it.
    if (result && !shortFold && (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") &&
        IsFloatForNaN(input.dataType)) {
        const bool isMax = reducerType == "stablehlo.maximum";
        auto i32 = [&](int v) { return [ctx.graph constantWithScalar:v dataType:MPSDataTypeInt32]; };
        MPSGraphTensor* isNan = [ctx.graph isNaNWithTensor:input name:nil];
        MPSGraphTensor* isInf = [ctx.graph isInfiniteWithTensor:input name:nil];
        MPSGraphTensor* positive =
            [ctx.graph greaterThanWithPrimaryTensor:input
                                    secondaryTensor:[ctx.graph constantWithScalar:0.0 dataType:input.dataType]
                                               name:nil];
        MPSGraphTensor* winSide = isMax ? positive : [ctx.graph notWithTensor:positive name:nil];
        MPSGraphTensor* winningInf = [ctx.graph logicalANDWithPrimaryTensor:isInf
                                                            secondaryTensor:winSide
                                                                       name:nil];
        // State is 2 * category, plus 1 for the winning zero (+0 for max, -0
        // for min): a prefix whose running max is 3 holds that zero, so a zero
        // result there is replaced by it (MPSGraph keeps one zero on a tie).
        MPSGraphTensor* state = [ctx.graph selectWithPredicateTensor:isInf
                                                 truePredicateTensor:i32(0)
                                                falsePredicateTensor:i32(2)
                                                                name:nil];
        const bool zeroTie = WindowZeroTieEnabled();
        if (zeroTie) {
            MPSGraphTensor* isZero =
                [ctx.graph equalWithPrimaryTensor:input
                                  secondaryTensor:[ctx.graph constantWithScalar:0.0 dataType:input.dataType]
                                             name:nil];
            MPSGraphTensor* neg = [ctx.graph signbitWithTensor:input name:nil];
            MPSGraphTensor* winSign = isMax ? [ctx.graph notWithTensor:neg name:nil] : neg;
            state = [ctx.graph selectWithPredicateTensor:[ctx.graph logicalANDWithPrimaryTensor:isZero
                                                                                secondaryTensor:winSign
                                                                                           name:nil]
                                     truePredicateTensor:i32(3)
                                    falsePredicateTensor:state
                                                    name:nil];
        }
        state = [ctx.graph selectWithPredicateTensor:winningInf
                                 truePredicateTensor:i32(4)
                                falsePredicateTensor:state
                                                name:nil];
        state = [ctx.graph selectWithPredicateTensor:isNan
                                 truePredicateTensor:i32(6)
                                falsePredicateTensor:state
                                                name:nil];
        MPSGraphTensor* seen = [ctx.graph cumulativeMaximumWithTensor:state axis:cumulativeAxis name:nil];
        const MPSDataType rt = result.dataType;
        auto when = [&](int s, double value, MPSGraphTensor* other) {
            MPSGraphTensor* hit = [ctx.graph equalWithPrimaryTensor:seen secondaryTensor:i32(s) name:nil];
            return [ctx.graph selectWithPredicateTensor:hit
                                    truePredicateTensor:[ctx.graph constantWithScalar:value dataType:rt]
                                   falsePredicateTensor:other
                                                   name:nil];
        };
        result = when(0, isMax ? -INFINITY : INFINITY, result);
        if (zeroTie) {
            MPSGraphTensor* hit3 = [ctx.graph equalWithPrimaryTensor:seen secondaryTensor:i32(3) name:nil];
            MPSGraphTensor* rz = [ctx.graph equalWithPrimaryTensor:result
                                                   secondaryTensor:[ctx.graph constantWithScalar:0.0
                                                                                        dataType:rt]
                                                              name:nil];
            result = [ctx.graph
                selectWithPredicateTensor:[ctx.graph logicalANDWithPrimaryTensor:hit3
                                                                 secondaryTensor:rz
                                                                            name:nil]
                      truePredicateTensor:[ctx.graph constantWithScalar:(isMax ? 0.0 : -0.0) dataType:rt]
                     falsePredicateTensor:result
                                     name:nil];
        }
        result = when(4, isMax ? INFINITY : -INFINITY, result);
        result = when(6, NAN, result);
    }
    if (!result) {
        return ProcessResult::Error("reduce_window: unsupported reducer for cumulative lowering (" +
                                    DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                windowStrides, baseDilations, windowDilations,
                                                                padding, reducerType) +
                                    ")");
    }

    if (result && reducerType == "stablehlo.add") {
        // A window smaller than the axis is a sliding sum, not a full
        // cumulative sum: subtract the prefix that has left the window.
        int64_t window = windowDimensions[static_cast<size_t>(cumulativeAxis)];
        int64_t axisSize = [input.shape[(NSUInteger)cumulativeAxis] longLongValue];
        if (window < axisSize) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
            for (size_t dim = 0; dim < rank; ++dim) {
                [starts addObject:@0];
                [ends addObject:dim == (size_t)cumulativeAxis ? @(axisSize - window) : input.shape[dim]];
                [strides addObject:@1];
                [left addObject:dim == (size_t)cumulativeAxis ? @(window) : @0];
                [right addObject:@0];
            }
            MPSGraphTensor* prefix = [ctx.graph sliceTensor:result
                                                     starts:starts
                                                       ends:ends
                                                    strides:strides
                                                       name:nil];
            MPSGraphTensor* shifted = [ctx.graph padTensor:prefix
                                           withPaddingMode:MPSGraphPaddingModeConstant
                                               leftPadding:left
                                              rightPadding:right
                                             constantValue:0.0
                                                      name:nil];
            result = [ctx.graph subtractionWithPrimaryTensor:result secondaryTensor:shifted name:nil];
        }
    }
    if (result && reverseCumulative && !shortFold) {
        result = [ctx.graph reverseTensor:result axes:axes name:nil];
    }
    if (result && (reducerType != "stablehlo.add" || !IsScalarZero(ctx.op->getOperand(1)))) {
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return ProcessResult::Error("reduce_window: missing init tensor");
        }
        if (reducerType == "stablehlo.add") {
            result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else if (reducerType == "stablehlo.multiply") {
            result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else if (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") {
            // Plain max/min replace a propagated NaN with init; keep it. A +-0
            // init ties with a zero prefix result, so the tie rule applies.
            result = WindowMaxMin(ctx.graph, result, init, reducerType == "stablehlo.maximum",
                                  ctx.op->getOperand(1));
        }
    }
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }
    return Result(ctx, result, "reduce_window");
}

// Window of one element per position along every axis (padding allowed).
static bool IsUnitStrideTrivialWindow(mlir::stablehlo::ReduceWindowOp op) {
    auto dims = op.getWindowDimensions();
    const size_t rank = dims.size();
    std::vector<int64_t> strides, baseDil, winDil;
    if (!ReadArrayOrDefaultOnes(op.getWindowStrides(), rank, strides) ||
        !ReadArrayOrDefaultOnes(op.getBaseDilations(), rank, baseDil) ||
        !ReadArrayOrDefaultOnes(op.getWindowDilations(), rank, winDil)) {
        return false;
    }
    for (size_t d = 0; d < rank; ++d) {
        if (dims[d] != 1 || strides[d] != 1 || baseDil[d] != 1 || winDil[d] != 1) {
            return false;
        }
    }
    return true;
}

// Bool mask (broadcastable to the output) of positions whose window covers no
// input element (only padding or base-dilation holes); nil when there are none.
static MPSGraphTensor* EmptyWindowMask(MPSGraph* g, mlir::stablehlo::ReduceWindowOp op) {
    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape()) {
        return nil;
    }
    auto dims = op.getWindowDimensions();
    const size_t rank = dims.size();
    std::vector<int64_t> strides, baseDil, winDil;
    std::vector<std::pair<int64_t, int64_t>> pad;
    if (static_cast<size_t>(outType.getRank()) != rank ||
        !ReadArrayOrDefaultOnes(op.getWindowStrides(), rank, strides) ||
        !ReadArrayOrDefaultOnes(op.getBaseDilations(), rank, baseDil) ||
        !ReadArrayOrDefaultOnes(op.getWindowDilations(), rank, winDil) ||
        !ReadPaddingOrDefaultZeros(op.getPadding(), rank, pad)) {
        return nil;
    }
    MPSGraphTensor* mask = nil;
    for (size_t d = 0; d < rank; ++d) {
        const int64_t n = inType.getDimSize(d);
        const int64_t outN = outType.getDimSize(d);
        if (n <= 0 || outN <= 0 || strides[d] <= 0 || baseDil[d] <= 0 || winDil[d] <= 0) {
            return nil;
        }
        const int64_t span = (n - 1) * baseDil[d];
        std::vector<uint8_t> empty(static_cast<size_t>(outN), 1);
        bool any = false;
        for (int64_t o = 0; o < outN; ++o) {
            const int64_t start = o * strides[d] - pad[d].first;
            // First window offset at or past 0; base-dilation residues repeat
            // every baseDil offsets, so that many probes decide the position.
            int64_t k = start >= 0 ? 0 : (-start + winDil[d] - 1) / winDil[d];
            for (int64_t tries = 0; k < dims[d] && tries < baseDil[d]; ++k, ++tries) {
                const int64_t p = start + k * winDil[d];
                if (p > span) {
                    break;
                }
                if (p % baseDil[d] == 0) {
                    empty[static_cast<size_t>(o)] = 0;
                    break;
                }
            }
            any = any || empty[static_cast<size_t>(o)] != 0;
        }
        if (!any) {
            continue;
        }
        NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:rank];
        for (size_t a = 0; a < rank; ++a) {
            [shape addObject:@(a == d ? outN : 1)];
        }
        NSData* data = [NSData dataWithBytes:empty.data() length:empty.size()];
        MPSGraphTensor* axisMask = [g castTensor:[g constantWithData:data
                                                               shape:shape
                                                            dataType:MPSDataTypeUInt8]
                                          toType:MPSDataTypeBool
                                            name:nil];
        mask = mask ? [g logicalORWithPrimaryTensor:mask secondaryTensor:axisMask name:nil] : axisMask;
    }
    return mask;
}

// Non-bool entry for f32 max/min windows, matching XLA:CPU bit for bit
// (build/probe_r20e_single_position_{flush,noflush}.log):
//  - a one-element stride-1 window with a constant identity init (-inf max,
//    +inf min) keeps subnormals, padded or not: lowered unchanged;
//  - otherwise the result is flush(reducer(flush(init), flush(x))) in the
//    -0 < +0 order: a subnormal init is fed as its signed zero (constant, so
//    GetInitZeroKind applies the tie rule) or flushed (runtime init), and the
//    result is flushed unless the float order-key path already did;
//  - a window covering only padding returns the raw init, unflushed.
static ProcessResult HandleReduceWindowFloat(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(ctx.op);
    if (!op || ctx.op->getNumOperands() != 2 || ctx.op->getNumResults() != 1 ||
        !WindowSubnormalFlushEnabled()) {
        return HandleReduceWindowImpl(ctx);
    }
    const std::string reducer = GetReductionOpType(op.getBody());
    const bool isMax = reducer == "stablehlo.maximum";
    if (!isMax && reducer != "stablehlo.minimum") {
        return HandleReduceWindowImpl(ctx);
    }
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* init = GetInputTensor(ctx, 1);
    if (!input || !init || input.dataType != MPSDataTypeFloat32 || init.dataType != MPSDataTypeFloat32) {
        return HandleReduceWindowImpl(ctx);
    }
    mlir::Value initValue = ctx.op->getOperand(1);
    if (IsF32InfScalar(initValue, isMax) && IsUnitStrideTrivialWindow(op)) {
        // XLA:CPU keeps subnormals here even when padded; the impl's padded
        // route flushed them (build/test_window_subnormal_flush_r20e3_flush.log),
        // so lower a non-negatively padded one as pad(max/min(x, init), init).
        const size_t rank = op.getWindowDimensions().size();
        std::vector<std::pair<int64_t, int64_t>> pad;
        if (!ReadPaddingOrDefaultZeros(op.getPadding(), rank, pad)) {
            return HandleReduceWindowImpl(ctx);
        }
        bool anyPad = false;
        bool negPad = false;
        NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
        for (size_t d = 0; d < rank; ++d) {
            anyPad = anyPad || pad[d].first != 0 || pad[d].second != 0;
            negPad = negPad || pad[d].first < 0 || pad[d].second < 0;
            [left addObject:@(pad[d].first)];
            [right addObject:@(pad[d].second)];
        }
        if (!anyPad || negPad || rank != input.shape.count) {
            return HandleReduceWindowImpl(ctx);
        }
        MPSGraphTensor* body = WindowMaxMin(ctx.graph, input, init, isMax, initValue);
        const double padValue = isMax ? -INFINITY : INFINITY;
        MPSGraphTensor* padded = [ctx.graph padTensor:body
                                      withPaddingMode:MPSGraphPaddingModeConstant
                                          leftPadding:left
                                         rightPadding:right
                                        constantValue:padValue
                                                 name:nil];
        NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
        if (outputShape && ![outputShape isEqualToArray:padded.shape]) {
            padded = [ctx.graph reshapeTensor:padded withShape:outputShape name:nil];
        }
        return Result(ctx, padded, "reduce_window");
    }
    bool subNeg = false;
    const bool constSub = IsF32SubnormalScalar(initValue, &subNeg);
    mlir::DenseElementsAttr dense;
    const bool initMayBeSub = constSub || !GetScalarConstantAttr(initValue, dense);

    void* initKey = initValue.getAsOpaquePointer();
    const bool initIsVariable = ctx.variables && ctx.variables->find(initKey) != ctx.variables->end();
    MPSGraphTensor* savedInit = nil;
    bool swapped = false;
    if (initMayBeSub && !initIsVariable && ctx.op->getOperand(0) != initValue) {
        auto it = ctx.values.find(initKey);
        if (it != ctx.values.end()) {
            savedInit = it->second;
            ctx.values[initKey] = constSub ? [ctx.graph constantWithScalar:(subNeg ? -0.0 : 0.0)
                                                                     shape:init.shape
                                                                  dataType:MPSDataTypeFloat32]
                                           : FlushF32SubnormalSigned(ctx.graph, init);
            swapped = true;
        }
    }
    bool skipFlush = false;
    ProcessResult r = HandleReduceWindowImpl(ctx, nullptr, &skipFlush);
    if (swapped) {
        ctx.values[initKey] = savedInit;
    }
    if (!r.ok()) {
        return r;
    }
    MPSGraphTensor* out = GetTensor(ctx.values, ctx.op->getResult(0));
    if (!out || out.dataType != MPSDataTypeFloat32) {
        return r;
    }
    if (!skipFlush) {
        out = FlushF32SubnormalSigned(ctx.graph, out);
    }
    if (initMayBeSub) {
        if (MPSGraphTensor* empty = EmptyWindowMask(ctx.graph, op)) {
            out = [ctx.graph selectWithPredicateTensor:empty
                                   truePredicateTensor:init
                                  falsePredicateTensor:out
                                                  name:nil];
        }
    }
    SetOutputTensor(ctx.values, ctx.op, out);
    return r;
}

// bool max/min windows (lax.reduce_window(x, False, lax.max, ...)) raised
// "unsupported pattern": every lowering path is numeric. Run them on the exact
// 0/1 uint8 widening through the integer extrema path and map back with != 0.
static ProcessResult HandleReduceWindow(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(ctx.op);
    if (!op || ctx.op->getNumOperands() != 2 || ctx.op->getNumResults() != 1 ||
        ctx.op->getOperand(0) == ctx.op->getOperand(1)) {
        return HandleReduceWindowFloat(ctx);
    }
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* init = GetInputTensor(ctx, 1);
    if (!input || !init || input.dataType != MPSDataTypeBool || init.dataType != MPSDataTypeBool) {
        return HandleReduceWindowFloat(ctx);
    }
    const std::string reducer = GetReductionOpType(op.getBody());
    // On 0/1 values or is max and and is min.
    const char* mapped = reducer == "stablehlo.maximum" || reducer == "stablehlo.or"    ? "stablehlo.maximum"
                         : reducer == "stablehlo.minimum" || reducer == "stablehlo.and" ? "stablehlo.minimum"
                                                                                        : nullptr;
    if (!mapped) {
        return HandleReduceWindowFloat(ctx);
    }
    void* inKey = ctx.op->getOperand(0).getAsOpaquePointer();
    void* initKey = ctx.op->getOperand(1).getAsOpaquePointer();
    if (ctx.variables && (ctx.variables->find(inKey) != ctx.variables->end() ||
                          ctx.variables->find(initKey) != ctx.variables->end())) {
        return HandleReduceWindowFloat(ctx);
    }
    auto savedIn = ctx.values.find(inKey);
    MPSGraphTensor* origIn = savedIn != ctx.values.end() ? savedIn->second : nil;
    auto savedInit = ctx.values.find(initKey);
    MPSGraphTensor* origInit = savedInit != ctx.values.end() ? savedInit->second : nil;
    ctx.values[inKey] = [ctx.graph castTensor:input toType:MPSDataTypeUInt8 name:nil];
    ctx.values[initKey] = [ctx.graph castTensor:init toType:MPSDataTypeUInt8 name:nil];
    ProcessResult r = HandleReduceWindowImpl(ctx, mapped);
    auto restoreKey = [&](void* key, MPSGraphTensor* orig) {
        if (orig) {
            ctx.values[key] = orig;
        } else {
            ctx.values.erase(key);
        }
    };
    restoreKey(inKey, origIn);
    restoreKey(initKey, origInit);
    if (!r.ok()) {
        return r;
    }
    MPSGraphTensor* widened = GetTensor(ctx.values, ctx.op->getResult(0));
    if (!widened) {
        return ProcessResult::Error("reduce_window: bool widening lowering failed");
    }
    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:widened.dataType];
    return Result(ctx, [ctx.graph notEqualWithPrimaryTensor:widened secondaryTensor:zero name:nil],
                  "reduce_window");
}
REGISTER_MPS_OP("stablehlo.reduce_window", HandleReduceWindow);

static ProcessResult HandleSelectAndScatter(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::SelectAndScatterOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("select_and_scatter: expected SelectAndScatterOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* source = GetInputTensor(ctx, 1);
    if (!operand || !source) {
        return ProcessResult::Error("select_and_scatter: missing input tensors");
    }

    enum class SelectMode { kUnknown, kMax, kMin };
    SelectMode selectMode = SelectMode::kUnknown;
    if (!op.getSelect().empty()) {
        for (mlir::Operation& nestedOp : op.getSelect().front()) {
            auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
            if (!compareOp) {
                continue;
            }
            auto dir = compareOp.getComparisonDirection();
            if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                dir == mlir::stablehlo::ComparisonDirection::GE) {
                selectMode = SelectMode::kMax;
            } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                       dir == mlir::stablehlo::ComparisonDirection::LE) {
                selectMode = SelectMode::kMin;
            }
            break;
        }
    }

    std::string scatterType = GetReductionOpType(op.getScatter());
    mlir::Value initValue = ctx.op->getOperand(2);

    if (selectMode == SelectMode::kUnknown || scatterType != "stablehlo.add" || !IsScalarZero(initValue)) {
#if MPS_LOG_LEVEL >= 3
        return ProcessResult::Error(
            "select_and_scatter: unsupported region bodies (expected select=GT/GE or LT/LE, "
            "scatter=add, init=0)");
#else
        return ProcessResult::Error("select_and_scatter: unsupported region bodies");
#endif
    }

    // Read window_dimensions / window_strides from the Properties struct first
    // (the OpaqueProperties path in newer StableHLO pins stores attrs as
    // typed property fields, not in the NamedAttrList). Fall back to the
    // NamedAttrList if Properties didn't yield values — the textual-IR path
    // emits attrs in the dictionary form which we then need to read from
    // getAttrOfType.
    std::vector<int64_t> windowDimensions;
    std::vector<int64_t> windowStrides;
    // DenseI64ArrayAttr exposes its data via asArrayRef() (no STL iterators).
    if (auto attr = op.getWindowDimensionsAttr()) {
        auto ar = attr.asArrayRef();
        windowDimensions.assign(ar.begin(), ar.end());
    } else if (auto attr = ctx.op->getAttrOfType<mlir::DenseI64ArrayAttr>("window_dimensions")) {
        auto ar = attr.asArrayRef();
        windowDimensions.assign(ar.begin(), ar.end());
    }
    if (auto attr = op.getWindowStridesAttr()) {
        auto ar = attr.asArrayRef();
        windowStrides.assign(ar.begin(), ar.end());
    } else if (auto attr = ctx.op->getAttrOfType<mlir::DenseI64ArrayAttr>("window_strides")) {
        auto ar = attr.asArrayRef();
        windowStrides.assign(ar.begin(), ar.end());
    }
    if (windowDimensions.empty()) {
        return ProcessResult::Error("select_and_scatter: missing window_dimensions");
    }
    if (windowStrides.empty()) {
        // Default to all-ones strides (matches the StableHLO spec default).
        windowStrides.assign(windowDimensions.size(), 1);
    }
    size_t rank = windowDimensions.size();
    bool isNHWC = (rank == 4 && windowStrides.size() == 4);
    bool isNWC = (rank == 3 && windowStrides.size() == 3);
    bool isSpatialOnly = (rank == 1 || rank == 2) && windowStrides.size() == rank;
    if (!isNHWC && !isNWC && !isSpatialOnly) {
        return ProcessResult::Error(
            "select_and_scatter: expected rank-1/2 spatial, rank-3 NWC, or rank-4 NHWC");
    }

    std::vector<std::pair<int64_t, int64_t>> padding;
    std::optional<mlir::DenseIntElementsAttr> paddingAttr;
    // Same OpaqueProperties fallback as for window_dimensions above.
    if (auto attr = op.getPaddingAttr()) {
        paddingAttr = attr;
    } else if (auto attr = ctx.op->getAttrOfType<mlir::DenseIntElementsAttr>("padding")) {
        paddingAttr = attr;
    }
    if (!ReadPaddingOrDefaultZeros(paddingAttr, rank, padding)) {
        return ProcessResult::Error("select_and_scatter: invalid padding attribute");
    }

    int64_t kH = 1, kW = 1;
    int64_t sH = 1, sW = 1;
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;

    if (isNHWC) {

        if (windowDimensions[0] != 1 || windowDimensions[3] != 1 || windowStrides[0] != 1 ||
            windowStrides[3] != 1 || padding[0].first != 0 || padding[0].second != 0 ||
            padding[3].first != 0 || padding[3].second != 0 || padding[1].first < 0 ||
            padding[1].second < 0 || padding[2].first < 0 || padding[2].second < 0) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error("select_and_scatter: unsupported NHWC pooling parameters");
#else
            return ProcessResult::Error("select_and_scatter: unsupported pooling parameters");
#endif
        }

        kH = windowDimensions[1];
        kW = windowDimensions[2];
        sH = windowStrides[1];
        sW = windowStrides[2];
        padTop = padding[1].first;
        padBottom = padding[1].second;
        padLeft = padding[2].first;
        padRight = padding[2].second;
    } else if (isNWC) {

        if (windowDimensions[0] != 1 || windowDimensions[2] != 1 || windowStrides[0] != 1 ||
            windowStrides[2] != 1 || padding[0].first != 0 || padding[0].second != 0 ||
            padding[2].first != 0 || padding[2].second != 0 || padding[1].first < 0 ||
            padding[1].second < 0) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error("select_and_scatter: unsupported NWC pooling parameters");
#else
            return ProcessResult::Error("select_and_scatter: unsupported pooling parameters");
#endif
        }

        if (!operand.shape || operand.shape.count != 3 || !source.shape || source.shape.count != 3) {
            return ProcessResult::Error("select_and_scatter: expected rank-3 NWC tensors");
        }
        operand = [ctx.graph reshapeTensor:operand
                                 withShape:@[operand.shape[0], @1, operand.shape[1], operand.shape[2]]
                                      name:nil];
        source = [ctx.graph reshapeTensor:source
                                withShape:@[source.shape[0], @1, source.shape[1], source.shape[2]]
                                     name:nil];

        kH = 1;
        kW = windowDimensions[1];
        sH = 1;
        sW = windowStrides[1];
        padTop = 0;
        padBottom = 0;
        padLeft = padding[1].first;
        padRight = padding[1].second;
    } else {
        if (!operand.shape || operand.shape.count != rank || !source.shape || source.shape.count != rank) {
            return ProcessResult::Error("select_and_scatter: rank-1/2 tensor shapes unavailable");
        }
        for (size_t dim = 0; dim < rank; ++dim) {
            if (windowDimensions[dim] <= 0 || windowStrides[dim] <= 0 || padding[dim].first < 0 ||
                padding[dim].second < 0) {
                return ProcessResult::Error("select_and_scatter: unsupported rank-1/2 pooling parameters");
            }
        }
        NSArray<NSNumber*>* operandShape =
            rank == 1 ? @[@1, @1, operand.shape[0], @1] : @[@1, operand.shape[0], operand.shape[1], @1];
        NSArray<NSNumber*>* sourceShape =
            rank == 1 ? @[@1, @1, source.shape[0], @1] : @[@1, source.shape[0], source.shape[1], @1];
        operand = [ctx.graph reshapeTensor:operand withShape:operandShape name:nil];
        source = [ctx.graph reshapeTensor:source withShape:sourceShape name:nil];
        size_t widthDim = rank - 1;
        kW = windowDimensions[widthDim];
        sW = windowStrides[widthDim];
        padLeft = padding[widthDim].first;
        padRight = padding[widthDim].second;
        if (rank == 2) {
            kH = windowDimensions[0];
            sH = windowStrides[0];
            padTop = padding[0].first;
            padBottom = padding[0].second;
        }
    }

    MPSGraphPooling2DOpDescriptor* desc =
        [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kW
                                                    kernelHeight:(NSUInteger)kH
                                                       strideInX:(NSUInteger)sW
                                                       strideInY:(NSUInteger)sH
                                                 dilationRateInX:1
                                                 dilationRateInY:1
                                                     paddingLeft:(NSUInteger)padLeft
                                                    paddingRight:(NSUInteger)padRight
                                                      paddingTop:(NSUInteger)padTop
                                                   paddingBottom:(NSUInteger)padBottom
                                                    paddingStyle:MPSGraphPaddingStyleExplicit
                                                      dataLayout:MPSGraphTensorNamedDataLayoutNHWC];

    MPSGraphTensor* pooledOperand = operand;
    if (selectMode == SelectMode::kMin) {
        pooledOperand = [ctx.graph negativeWithTensor:operand name:nil];
    }

    MPSGraphTensor* result = [ctx.graph maxPooling2DGradientWithGradientTensor:source
                                                                  sourceTensor:pooledOperand
                                                                    descriptor:desc
                                                                          name:nil];
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }
    return Result(ctx, result, "select_and_scatter");
}
REGISTER_MPS_OP("stablehlo.select_and_scatter", HandleSelectAndScatter);

static ProcessResult HandleReturn(HandlerContext& ctx) {

    MPS_LOG_DEBUG("stablehlo.return terminator encountered; skipping execution\n");
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.return", HandleReturn);

static ProcessResult HandleBatchNormInference(HandlerContext& ctx) {
    auto bnOp = mlir::dyn_cast<mlir::stablehlo::BatchNormInferenceOp>(ctx.op);
    if (!bnOp)
        return ProcessResult::Error("batch_norm_inference: expected BatchNormInferenceOp");

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* scale = GetInputTensor(ctx, 1);
    MPSGraphTensor* offset = GetInputTensor(ctx, 2);
    MPSGraphTensor* mean = GetInputTensor(ctx, 3);
    MPSGraphTensor* variance = GetInputTensor(ctx, 4);
    if (!operand || !scale || !offset || !mean || !variance)
        return ProcessResult::Error("batch_norm_inference: missing input tensor");

    float epsilon = bnOp.getEpsilon().convertToFloat();
    int64_t feature_index = bnOp.getFeatureIndex();

    NSUInteger rank = operand.shape.count;
    NSMutableArray<NSNumber*>* bcast_shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger i = 0; i < rank; i++) {
        [bcast_shape addObject:(i == (NSUInteger)feature_index ? operand.shape[i] : @1)];
    }

    scale = [ctx.graph reshapeTensor:scale withShape:bcast_shape name:nil];
    offset = [ctx.graph reshapeTensor:offset withShape:bcast_shape name:nil];
    mean = [ctx.graph reshapeTensor:mean withShape:bcast_shape name:nil];
    variance = [ctx.graph reshapeTensor:variance withShape:bcast_shape name:nil];

    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsilon dataType:operand.dataType];
    MPSGraphTensor* var_eps = [ctx.graph additionWithPrimaryTensor:variance secondaryTensor:eps name:nil];
    MPSGraphTensor* inv_std = [ctx.graph reciprocalSquareRootWithTensor:var_eps name:nil];
    MPSGraphTensor* centered = [ctx.graph subtractionWithPrimaryTensor:operand secondaryTensor:mean name:nil];
    MPSGraphTensor* normalized = [ctx.graph multiplicationWithPrimaryTensor:centered
                                                            secondaryTensor:inv_std
                                                                       name:nil];
    MPSGraphTensor* scaled = [ctx.graph multiplicationWithPrimaryTensor:normalized
                                                        secondaryTensor:scale
                                                                   name:nil];
    MPSGraphTensor* result = [ctx.graph additionWithPrimaryTensor:scaled secondaryTensor:offset name:nil];

    return Result(ctx, result, "batch_norm_inference");
}
REGISTER_MPS_OP("stablehlo.batch_norm_inference", HandleBatchNormInference);

static ProcessResult HandleBatchNormTraining(HandlerContext& ctx) {
    auto bnOp = mlir::dyn_cast<mlir::stablehlo::BatchNormTrainingOp>(ctx.op);
    if (!bnOp)
        return ProcessResult::Error("batch_norm_training: expected BatchNormTrainingOp");

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* scale = GetInputTensor(ctx, 1);
    MPSGraphTensor* offset = GetInputTensor(ctx, 2);
    if (!operand || !scale || !offset)
        return ProcessResult::Error("batch_norm_training: missing input tensor");

    float epsilon = bnOp.getEpsilon().convertToFloat();
    int64_t feature_index = bnOp.getFeatureIndex();

    NSUInteger rank = operand.shape.count;

    NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        if (i != (NSUInteger)feature_index) {
            [reduce_axes addObject:@(i)];
        }
    }

    MPSGraphTensor* batch_mean = [ctx.graph meanOfTensor:operand axes:reduce_axes name:nil];
    MPSGraphTensor* batch_var = [ctx.graph varianceOfTensor:operand axes:reduce_axes name:nil];

    NSMutableArray<NSNumber*>* bcast_shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger i = 0; i < rank; i++) {
        [bcast_shape addObject:(i == (NSUInteger)feature_index ? operand.shape[i] : @1)];
    }

    MPSGraphTensor* mean_bcast = [ctx.graph reshapeTensor:batch_mean withShape:bcast_shape name:nil];
    MPSGraphTensor* var_bcast = [ctx.graph reshapeTensor:batch_var withShape:bcast_shape name:nil];
    MPSGraphTensor* scale_bcast = [ctx.graph reshapeTensor:scale withShape:bcast_shape name:nil];
    MPSGraphTensor* offset_bcast = [ctx.graph reshapeTensor:offset withShape:bcast_shape name:nil];

    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsilon dataType:operand.dataType];
    MPSGraphTensor* var_eps = [ctx.graph additionWithPrimaryTensor:var_bcast secondaryTensor:eps name:nil];
    MPSGraphTensor* inv_std = [ctx.graph reciprocalSquareRootWithTensor:var_eps name:nil];
    MPSGraphTensor* centered = [ctx.graph subtractionWithPrimaryTensor:operand
                                                       secondaryTensor:mean_bcast
                                                                  name:nil];
    MPSGraphTensor* normalized = [ctx.graph multiplicationWithPrimaryTensor:centered
                                                            secondaryTensor:inv_std
                                                                       name:nil];
    MPSGraphTensor* scaled = [ctx.graph multiplicationWithPrimaryTensor:normalized
                                                        secondaryTensor:scale_bcast
                                                                   name:nil];
    MPSGraphTensor* output = [ctx.graph additionWithPrimaryTensor:scaled
                                                  secondaryTensor:offset_bcast
                                                             name:nil];

    batch_mean = [ctx.graph reshapeTensor:batch_mean
                                withShape:@[operand.shape[(NSUInteger)feature_index]]
                                     name:nil];
    batch_var = [ctx.graph reshapeTensor:batch_var
                               withShape:@[operand.shape[(NSUInteger)feature_index]]
                                    name:nil];

    SetOutputTensor(ctx.values, ctx.op, output, 0);
    SetOutputTensor(ctx.values, ctx.op, batch_mean, 1);
    SetOutputTensor(ctx.values, ctx.op, batch_var, 2);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.batch_norm_training", HandleBatchNormTraining);

static ProcessResult HandleBatchNormGrad(HandlerContext& ctx) {
    auto bnOp = mlir::dyn_cast<mlir::stablehlo::BatchNormGradOp>(ctx.op);
    if (!bnOp)
        return ProcessResult::Error("batch_norm_grad: expected BatchNormGradOp");

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* scale = GetInputTensor(ctx, 1);
    MPSGraphTensor* mean = GetInputTensor(ctx, 2);
    MPSGraphTensor* variance = GetInputTensor(ctx, 3);
    MPSGraphTensor* grad_output = GetInputTensor(ctx, 4);
    if (!operand || !scale || !mean || !variance || !grad_output)
        return ProcessResult::Error("batch_norm_grad: missing input tensor");

    float epsilon = bnOp.getEpsilon().convertToFloat();
    int64_t feature_index = bnOp.getFeatureIndex();

    NSUInteger rank = operand.shape.count;

    NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        if (i != (NSUInteger)feature_index) {
            [reduce_axes addObject:@(i)];
        }
    }

    double N = 1.0;
    for (NSUInteger i = 0; i < rank; i++) {
        if (i != (NSUInteger)feature_index) {
            N *= [operand.shape[i] longLongValue];
        }
    }

    NSMutableArray<NSNumber*>* bcast_shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger i = 0; i < rank; i++) {
        [bcast_shape addObject:(i == (NSUInteger)feature_index ? operand.shape[i] : @1)];
    }

    MPSGraphTensor* mean_bcast = [ctx.graph reshapeTensor:mean withShape:bcast_shape name:nil];
    MPSGraphTensor* var_bcast = [ctx.graph reshapeTensor:variance withShape:bcast_shape name:nil];
    MPSGraphTensor* scale_bcast = [ctx.graph reshapeTensor:scale withShape:bcast_shape name:nil];

    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsilon dataType:operand.dataType];
    MPSGraphTensor* var_eps = [ctx.graph additionWithPrimaryTensor:var_bcast secondaryTensor:eps name:nil];
    MPSGraphTensor* inv_std = [ctx.graph reciprocalSquareRootWithTensor:var_eps name:nil];
    MPSGraphTensor* centered = [ctx.graph subtractionWithPrimaryTensor:operand
                                                       secondaryTensor:mean_bcast
                                                                  name:nil];
    MPSGraphTensor* x_hat = [ctx.graph multiplicationWithPrimaryTensor:centered
                                                       secondaryTensor:inv_std
                                                                  name:nil];

    MPSGraphTensor* grad_offset = [ctx.graph reductionSumWithTensor:grad_output axes:reduce_axes name:nil];
    grad_offset = [ctx.graph reshapeTensor:grad_offset
                                 withShape:@[operand.shape[(NSUInteger)feature_index]]
                                      name:nil];

    MPSGraphTensor* grad_x_hat = [ctx.graph multiplicationWithPrimaryTensor:grad_output
                                                            secondaryTensor:x_hat
                                                                       name:nil];
    MPSGraphTensor* grad_scale = [ctx.graph reductionSumWithTensor:grad_x_hat axes:reduce_axes name:nil];
    grad_scale = [ctx.graph reshapeTensor:grad_scale
                                withShape:@[operand.shape[(NSUInteger)feature_index]]
                                     name:nil];

    MPSGraphTensor* n_const = [ctx.graph constantWithScalar:N dataType:operand.dataType];
    MPSGraphTensor* grad_scale_bcast = [ctx.graph reshapeTensor:grad_scale withShape:bcast_shape name:nil];
    MPSGraphTensor* grad_offset_bcast = [ctx.graph reshapeTensor:grad_offset withShape:bcast_shape name:nil];

    MPSGraphTensor* term1 = [ctx.graph multiplicationWithPrimaryTensor:x_hat
                                                       secondaryTensor:grad_scale_bcast
                                                                  name:nil];
    MPSGraphTensor* term2 = [ctx.graph additionWithPrimaryTensor:term1
                                                 secondaryTensor:grad_offset_bcast
                                                            name:nil];
    MPSGraphTensor* term3 = [ctx.graph divisionWithPrimaryTensor:term2 secondaryTensor:n_const name:nil];
    MPSGraphTensor* grad_normed = [ctx.graph subtractionWithPrimaryTensor:grad_output
                                                          secondaryTensor:term3
                                                                     name:nil];
    MPSGraphTensor* grad_operand = [ctx.graph multiplicationWithPrimaryTensor:grad_normed
                                                              secondaryTensor:scale_bcast
                                                                         name:nil];
    grad_operand = [ctx.graph multiplicationWithPrimaryTensor:grad_operand secondaryTensor:inv_std name:nil];

    SetOutputTensor(ctx.values, ctx.op, grad_operand, 0);
    SetOutputTensor(ctx.values, ctx.op, grad_scale, 1);
    SetOutputTensor(ctx.values, ctx.op, grad_offset, 2);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.batch_norm_grad", HandleBatchNormGrad);

} // namespace jax_metallib
