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

#include "pjrt_plugin/core/stablehlo_parser.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <unordered_set>

namespace mps {

namespace {

const std::unordered_set<std::string>& getSupportedOps() {
    static std::unordered_set<std::string> supported = []() {
        auto ops = jax_metallib::OpRegistry::GetRegisteredOps();

        // The pre-parse walk in checkUnsupportedOps traverses every op in the
        // module, including those inside composite decomposition function bodies.
        // jax 0.10 lowers many higher-level ops (jnp.arcsin, jnp.argsort, …) as
        // stablehlo.composite, which processCompositeOp handles at runtime by
        // walking the referenced decomposition. So composite itself, plus the
        // stablehlo.return terminator that closes a decomposition body, must
        // pass the pre-parse check.
        ops.insert("func.return");
        ops.insert("func.call");
        ops.insert("stablehlo.composite");
        ops.insert("stablehlo.return");
        return ops;
    }();
    return supported;
}

void registerDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    // Without the func inliner extension, createInlinerPass() cannot inline
    // func.call, so jitted functions kept top-level calls (e.g. @lu_solve) and
    // native ops inside them (triangular_solve) reached the recursive graph
    // builder and failed. JAX_METALLIB_DISABLE_FUNC_INLINER=1 restores that.
    const char* noInline = std::getenv("JAX_METALLIB_DISABLE_FUNC_INLINER");
    if (!(noInline && noInline[0] != '\0' && !(noInline[0] == '0' && noInline[1] == '\0'))) {
        mlir::func::registerInlinerExtension(registry);
    }
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    context.allowUnregisteredDialects();
}

bool runInlinerPass(mlir::MLIRContext& context, mlir::ModuleOp module) {

    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() != "main" && funcOp.isPublic()) {
            funcOp.setPrivate();
        }
    });

    mlir::PassManager pm(&context);
    pm.addPass(mlir::createInlinerPass());

    (void)pm.run(module);
    return true;
}

mlir::func::FuncOp findEntryFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp entry = nullptr;

    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() == "main" || !entry) {
            entry = funcOp;
        }
    });

    return entry;
}

// lax.scan over the layers of a model lowers to a counted while whose body
// indexes stacked weights with dynamic_slice(w, layer_index). In the graph
// that index is a runtime tensor, so every slice is clamped with
// compare/select and fed to sliceTensor:startTensor:. GPT-2 medium f32
// (6 layers) took 5.9-6.3 ms as a scan and also 6.30 ms when the MPSGraph
// loop was unrolled in-graph, vs 5.06-5.49 ms unrolled in Python (static
// slices). A top-level counted while with a constant lower bound, constant
// upper bound, a positive constant step and at most
// JAX_METALLIB_LOOP_PARSE_UNROLL_MAX_TRIPS trips (default 16; 0 disables) is
// therefore unrolled here: the body is cloned per trip with the counter
// replaced by a constant. Only in the cloned ops, scalar integer
// add/sub/mul/compare/select/reshape and dynamic_slice of an iota or small
// constant are then folded exactly (two's-complement wrap, XLA clamping), and
// a dynamic_slice whose starts are all constant becomes a plain slice. Bodies
// with nested control flow, buffer writes, calls, custom calls or any op with
// memory effects are left alone, and the unrolled size is capped at 4096 ops.
int64_t parseUnrollMaxTrips() {
    static const int64_t value = [] {
        const char* v = std::getenv("JAX_METALLIB_LOOP_PARSE_UNROLL_MAX_TRIPS");
        if (!v || v[0] == '\0') {
            return static_cast<int64_t>(16);
        }
        long k = std::strtol(v, nullptr, 10);
        return static_cast<int64_t>(std::max(0L, std::min(k, 64L)));
    }();
    return value;
}

bool tracePlanEnabled() {
    const char* v = std::getenv("JAX_METALLIB_TRACE_PLAN");
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

// Rank-0 integer tensor type with width in [minWidth, 64], else null.
mlir::RankedTensorType intScalarType(mlir::Type t, unsigned minWidth) {
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t);
    if (!rt || rt.getRank() != 0) {
        return {};
    }
    auto it = mlir::dyn_cast<mlir::IntegerType>(rt.getElementType());
    if (!it || it.getWidth() < minWidth || it.getWidth() > 64) {
        return {};
    }
    return rt;
}

bool readIntScalar(mlir::Value v, llvm::APInt* out) {
    auto c = v.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!c) {
        return false;
    }
    auto attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(c.getValue());
    if (!attr || attr.getNumElements() != 1 || !mlir::isa<mlir::IntegerType>(attr.getElementType())) {
        return false;
    }
    *out = *attr.getValues<llvm::APInt>().begin();
    return true;
}

mlir::Value makeIntConst(mlir::OpBuilder& b, mlir::Location loc, mlir::RankedTensorType type,
                         llvm::ArrayRef<llvm::APInt> values) {
    auto attr = mlir::DenseElementsAttr::get(type, values);
    return b.create<mlir::stablehlo::ConstantOp>(loc, attr).getResult();
}

struct ParseCountedLoop {
    unsigned counter = 0;
    int64_t lower = 0;
    int64_t step = 1;
    int64_t trips = 0;
    mlir::RankedTensorType type;
};

bool matchParseCountedLoop(mlir::stablehlo::WhileOp w, ParseCountedLoop* out) {
    namespace shlo = mlir::stablehlo;
    if (w.getCond().empty() || w.getBody().empty()) {
        return false;
    }
    mlir::Block& cond = w.getCond().front();
    mlir::Block& body = w.getBody().front();
    const unsigned n = w->getNumOperands();
    if (cond.getNumArguments() != n || body.getNumArguments() != n || w->getNumResults() != n) {
        return false;
    }
    auto condRet = mlir::dyn_cast<shlo::ReturnOp>(cond.getTerminator());
    if (!condRet || condRet->getNumOperands() != 1) {
        return false;
    }
    auto cmp = condRet->getOperand(0).getDefiningOp<shlo::CompareOp>();
    if (!cmp || cmp.getComparisonDirection() != shlo::ComparisonDirection::LT) {
        return false;
    }
    if (auto ct = cmp.getCompareType(); ct && *ct != shlo::ComparisonType::SIGNED) {
        return false;
    }
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(cmp.getLhs());
    llvm::APInt upper;
    if (!arg || arg.getOwner() != &cond || !readIntScalar(cmp.getRhs(), &upper)) {
        return false;
    }
    const unsigned idx = arg.getArgNumber();
    auto type = intScalarType(w->getOperand(idx).getType(), 8);
    if (!type || mlir::cast<mlir::IntegerType>(type.getElementType()).isUnsigned()) {
        return false;
    }
    llvm::APInt lower;
    llvm::APInt step;
    if (!readIntScalar(w->getOperand(idx), &lower)) {
        return false;
    }
    auto bodyRet = mlir::dyn_cast<shlo::ReturnOp>(body.getTerminator());
    if (!bodyRet || bodyRet->getNumOperands() != n) {
        return false;
    }
    auto add = bodyRet->getOperand(idx).getDefiningOp<shlo::AddOp>();
    if (!add) {
        return false;
    }
    mlir::Value counterArg = body.getArgument(idx);
    mlir::Value stepVal;
    if (add.getLhs() == counterArg) {
        stepVal = add.getRhs();
    } else if (add.getRhs() == counterArg) {
        stepVal = add.getLhs();
    }
    if (!stepVal || !readIntScalar(stepVal, &step)) {
        return false;
    }
    const unsigned width = mlir::cast<mlir::IntegerType>(type.getElementType()).getWidth();
    const int64_t lo = lower.getSExtValue();
    const int64_t hi = upper.getSExtValue();
    const int64_t st = step.getSExtValue();
    if (st <= 0 || hi <= lo) {
        return false;
    }
    // The last counter value (< hi + st) must not wrap, or the real loop
    // would keep running past the trip count computed here.
    const int64_t maxVal = width >= 64 ? INT64_MAX : (int64_t{1} << (width - 1)) - 1;
    int64_t lastPlus = 0;
    if (__builtin_add_overflow(hi - 1, st, &lastPlus) || lastPlus > maxVal) {
        return false;
    }
    out->counter = idx;
    out->lower = lo;
    out->step = st;
    out->trips = (hi - lo + st - 1) / st;
    out->type = type;
    return true;
}

bool isParseUnrollableBody(mlir::Block& body, int64_t trips) {
    namespace shlo = mlir::stablehlo;
    int64_t numOps = 0;
    for (mlir::Operation& op : body.without_terminator()) {
        ++numOps;
        if (!mlir::isMemoryEffectFree(&op)) {
            return false;
        }
    }
    bool ok = true;
    body.walk([&](mlir::Operation* op) {
        if (mlir::isa<shlo::WhileOp, shlo::CaseOp, shlo::IfOp, shlo::DynamicUpdateSliceOp, shlo::ScatterOp,
                      shlo::CustomCallOp, mlir::func::CallOp>(op)) {
            ok = false;
        }
    });
    return ok && numOps * trips <= 4096;
}

void unrollParseCountedLoop(mlir::stablehlo::WhileOp w, const ParseCountedLoop& c,
                            llvm::DenseSet<mlir::Operation*>& owned) {
    mlir::Block& body = w.getBody().front();
    mlir::Operation* ret = body.getTerminator();
    mlir::OpBuilder b(w);
    llvm::SmallVector<mlir::Value> cur(w->getOperands().begin(), w->getOperands().end());
    const unsigned width = mlir::cast<mlir::IntegerType>(c.type.getElementType()).getWidth();
    for (int64_t t = 0; t < c.trips; ++t) {
        mlir::IRMapping map;
        llvm::APInt value(width, static_cast<uint64_t>(c.lower + t * c.step), /*isSigned=*/true);
        mlir::Value counter = makeIntConst(b, w.getLoc(), c.type, {value});
        owned.insert(counter.getDefiningOp());
        for (unsigned i = 0; i < body.getNumArguments(); ++i) {
            map.map(body.getArgument(i), i == c.counter ? counter : cur[i]);
        }
        for (mlir::Operation& op : body.without_terminator()) {
            owned.insert(b.clone(op, map));
        }
        for (unsigned i = 0; i < ret->getNumOperands(); ++i) {
            cur[i] = map.lookupOrDefault(ret->getOperand(i));
        }
    }
    w->replaceAllUsesWith(cur);
    w->erase();
}

// One folding sweep over the cloned ops; returns true when anything changed.
bool foldUnrolledIndexMath(mlir::Block& block, llvm::DenseSet<mlir::Operation*>& owned) {
    namespace shlo = mlir::stablehlo;
    bool changed = false;
    for (mlir::Operation& opRef : llvm::make_early_inc_range(block)) {
        mlir::Operation* op = &opRef;
        if (!owned.count(op) || op->getNumResults() != 1) {
            continue;
        }
        mlir::OpBuilder b(op);
        mlir::Value result = op->getResult(0);
        mlir::Value replacement;

        if (auto sel = mlir::dyn_cast<shlo::SelectOp>(op)) {
            llvm::APInt p;
            if (intScalarType(sel.getPred().getType(), 1) && readIntScalar(sel.getPred(), &p)) {
                mlir::Value chosen = p.isZero() ? sel.getOnFalse() : sel.getOnTrue();
                if (chosen.getType() == result.getType()) {
                    replacement = chosen;
                }
            }
        } else if (mlir::isa<shlo::AddOp, shlo::SubtractOp, shlo::MulOp>(op)) {
            auto type = intScalarType(result.getType(), 2);
            llvm::APInt l;
            llvm::APInt r;
            if (type && readIntScalar(op->getOperand(0), &l) && readIntScalar(op->getOperand(1), &r) &&
                l.getBitWidth() == r.getBitWidth()) {
                llvm::APInt v = mlir::isa<shlo::AddOp>(op)        ? l + r
                                : mlir::isa<shlo::SubtractOp>(op) ? l - r
                                                                  : l * r;
                replacement = makeIntConst(b, op->getLoc(), type, {v});
            }
        } else if (auto cmp = mlir::dyn_cast<shlo::CompareOp>(op)) {
            auto lt = intScalarType(cmp.getLhs().getType(), 2);
            auto rtype = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
            llvm::APInt l;
            llvm::APInt r;
            auto ct = cmp.getCompareType();
            const bool unsignedElem = lt && mlir::cast<mlir::IntegerType>(lt.getElementType()).isUnsigned();
            const bool isSigned = (!ct || *ct == shlo::ComparisonType::SIGNED) && !unsignedElem;
            const bool isUnsigned = (ct && *ct == shlo::ComparisonType::UNSIGNED) || (!ct && unsignedElem);
            if (lt && rtype && rtype.getRank() == 0 && (isSigned || isUnsigned) &&
                readIntScalar(cmp.getLhs(), &l) && readIntScalar(cmp.getRhs(), &r) &&
                l.getBitWidth() == r.getBitWidth()) {
                bool v = false;
                bool known = true;
                switch (cmp.getComparisonDirection()) {
                case shlo::ComparisonDirection::EQ:
                    v = l == r;
                    break;
                case shlo::ComparisonDirection::NE:
                    v = l != r;
                    break;
                case shlo::ComparisonDirection::LT:
                    v = isSigned ? l.slt(r) : l.ult(r);
                    break;
                case shlo::ComparisonDirection::LE:
                    v = isSigned ? l.sle(r) : l.ule(r);
                    break;
                case shlo::ComparisonDirection::GT:
                    v = isSigned ? l.sgt(r) : l.ugt(r);
                    break;
                case shlo::ComparisonDirection::GE:
                    v = isSigned ? l.sge(r) : l.uge(r);
                    break;
                default:
                    known = false;
                }
                if (known) {
                    replacement = makeIntConst(b, op->getLoc(), rtype, {llvm::APInt(1, v ? 1 : 0)});
                }
            }
        } else if (mlir::isa<shlo::ReshapeOp>(op)) {
            auto c = op->getOperand(0).getDefiningOp<shlo::ConstantOp>();
            auto rtype = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
            auto attr = c ? mlir::dyn_cast<mlir::DenseIntElementsAttr>(c.getValue()) : nullptr;
            if (attr && rtype && rtype.hasStaticShape() && attr.getNumElements() <= 64) {
                auto reshaped = attr.reshape(rtype);
                replacement = b.create<shlo::ConstantOp>(op->getLoc(), reshaped).getResult();
            }
        } else if (auto ds = mlir::dyn_cast<shlo::DynamicSliceOp>(op)) {
            mlir::Value operand = op->getOperand(0);
            auto otype = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
            auto rtype = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
            llvm::ArrayRef<int64_t> sizes = ds.getSliceSizes();
            const unsigned rank = otype ? static_cast<unsigned>(otype.getRank()) : 0;
            bool ok = otype && rtype && otype.hasStaticShape() && rtype.hasStaticShape() &&
                      sizes.size() == rank && op->getNumOperands() == rank + 1;
            llvm::SmallVector<int64_t> starts;
            llvm::SmallVector<int64_t> limits;
            for (unsigned d = 0; ok && d < rank; ++d) {
                llvm::APInt s;
                if (!readIntScalar(op->getOperand(d + 1), &s)) {
                    ok = false;
                    break;
                }
                const int64_t maxStart = otype.getDimSize(d) - sizes[d];
                if (maxStart < 0) {
                    ok = false;
                    break;
                }
                const int64_t clamped = std::min<int64_t>(std::max<int64_t>(s.getSExtValue(), 0), maxStart);
                starts.push_back(clamped);
                limits.push_back(clamped + sizes[d]);
            }
            if (ok) {
                auto elemInt = mlir::dyn_cast<mlir::IntegerType>(otype.getElementType());
                auto iota = operand.getDefiningOp<shlo::IotaOp>();
                auto cst = operand.getDefiningOp<shlo::ConstantOp>();
                auto cattr = cst ? mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue()) : nullptr;
                if (rank == 1 && elemInt && rtype.getNumElements() <= 64 && (iota || cattr)) {
                    llvm::SmallVector<llvm::APInt> vals;
                    for (int64_t k = starts[0]; k < limits[0]; ++k) {
                        if (iota) {
                            vals.push_back(llvm::APInt(elemInt.getWidth(), static_cast<uint64_t>(k), true));
                        } else {
                            vals.push_back(cattr.getValues<llvm::APInt>()[k]);
                        }
                    }
                    replacement = makeIntConst(b, op->getLoc(), rtype, vals);
                } else {
                    llvm::SmallVector<int64_t> strides(rank, 1);
                    auto slice = b.create<shlo::SliceOp>(op->getLoc(), operand, starts, limits, strides);
                    if (slice.getResult().getType() == result.getType()) {
                        replacement = slice.getResult();
                    } else {
                        slice->erase();
                    }
                }
            }
        }

        if (!replacement) {
            continue;
        }
        if (mlir::Operation* def = replacement.getDefiningOp();
            def && def->getBlock() == &block && def->isBeforeInBlock(op) && !owned.count(def)) {
            // A value chosen by select may be defined outside the cloned ops;
            // only newly created ops become owned.
            if (def->getLoc() == op->getLoc() && def->getNextNode() == op) {
                owned.insert(def);
            }
        }
        result.replaceAllUsesWith(replacement);
        owned.erase(op);
        op->erase();
        changed = true;
    }
    return changed;
}

void unrollShortCountedLoops(mlir::func::FuncOp entry) {
    const int64_t maxTrips = parseUnrollMaxTrips();
    if (maxTrips <= 0 || !entry || entry.getBody().empty()) {
        return;
    }
    mlir::Block& block = entry.getBody().front();
    llvm::SmallVector<mlir::stablehlo::WhileOp> loops;
    for (mlir::Operation& op : block) {
        if (auto w = mlir::dyn_cast<mlir::stablehlo::WhileOp>(op)) {
            loops.push_back(w);
        }
    }
    const bool trace = tracePlanEnabled();
    llvm::DenseSet<mlir::Operation*> owned;
    int unrolled = 0;
    for (auto w : loops) {
        ParseCountedLoop c;
        const bool matched = matchParseCountedLoop(w, &c);
        const bool ok = matched && c.trips <= maxTrips && isParseUnrollableBody(w.getBody().front(), c.trips);
        if (trace) {
            fprintf(stderr,
                    "JAX_METALLIB_PLAN_JSON:{\"event\":\"parse_loop_unroll\",\"counted\":%s,\"trips\":%lld,"
                    "\"unrolled\":%s}\n",
                    matched ? "true" : "false", (long long)(matched ? c.trips : -1), ok ? "true" : "false");
        }
        if (!ok) {
            continue;
        }
        unrollParseCountedLoop(w, c, owned);
        ++unrolled;
    }
    if (unrolled == 0) {
        return;
    }
    for (int i = 0; i < 64 && foldUnrolledIndexMath(block, owned); ++i) {
    }
    bool erased = true;
    while (erased) {
        erased = false;
        for (mlir::Operation& op : llvm::make_early_inc_range(llvm::reverse(block))) {
            if (&op != block.getTerminator() && owned.count(&op) && mlir::isOpTriviallyDead(&op)) {
                owned.erase(&op);
                op.erase();
                erased = true;
            }
        }
    }
    if (trace) {
        int dynSlices = 0;
        for (mlir::Operation& op : block) {
            dynSlices += mlir::isa<mlir::stablehlo::DynamicSliceOp>(op) ? 1 : 0;
        }
        fprintf(stderr,
                "JAX_METALLIB_PLAN_JSON:{\"event\":\"parse_loop_unroll_done\",\"loops\":%d,"
                "\"remaining_dynamic_slices\":%d}\n",
                unrolled, dynSlices);
    }
}

std::vector<std::string> checkUnsupportedOps(mlir::ModuleOp module) {
    std::unordered_set<std::string> unsupported_set;
    const auto& supported = getSupportedOps();

    module.walk([&](mlir::Operation* op) {
        if (mlir::isa<mlir::ModuleOp>(op) || mlir::isa<mlir::func::FuncOp>(op)) {
            return;
        }

        std::string name = op->getName().getStringRef().str();
        if (supported.find(name) == supported.end()) {
            unsupported_set.insert(name);
        }
    });

    return std::vector<std::string>(unsupported_set.begin(), unsupported_set.end());
}

ParsedModule finalizeModule(std::unique_ptr<mlir::MLIRContext> context,
                            mlir::OwningOpRef<mlir::ModuleOp> module) {
    ParsedModule result;

    if (!module) {
        return result;
    }

    if (!runInlinerPass(*context, *module)) {
        return result;
    }

    mlir::func::FuncOp entry = findEntryFunction(*module);
    if (!entry) {
        return result;
    }
    unrollShortCountedLoops(entry);

    result.unsupported_ops = checkUnsupportedOps(*module);

    result.context = std::move(context);
    result.module = std::move(module);
    result.entry_func = entry;

    return result;
}

} // namespace

ParsedModule parseStableHLOBytecode(const char* data, size_t size) {
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size), "stablehlo_bytecode", false);

    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), context.get());

    if (!moduleOp) {

        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    }

    return finalizeModule(std::move(context), std::move(moduleOp));
}

ParsedModule parseStableHLOText(const std::string& text) {
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    mlir::OwningOpRef<mlir::ModuleOp> moduleOp = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());

    return finalizeModule(std::move(context), std::move(moduleOp));
}

std::string bytecodeToText(const char* data, size_t size) {
    mlir::MLIRContext context;
    registerDialects(context);

    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size), "stablehlo_bytecode", false);

    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context);

    if (!moduleOp) {

        llvm::SourceMgr sourceMgr;
        auto bufferCopy =
            llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size), "stablehlo_bytecode", false);
        sourceMgr.AddNewSourceBuffer(std::move(bufferCopy), llvm::SMLoc());
        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    }

    if (!moduleOp) {
        return "";
    }

    std::string result;
    llvm::raw_string_ostream os(result);
    moduleOp->print(os);
    return result;
}

} // namespace mps
