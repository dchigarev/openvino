// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "openvino/op/op.hpp"
#include "openvino/core/shape.hpp"

#include "convert_common.hpp"

#if defined(GRAPH_COMPILER) && defined(GC_ENABLE_IMEX)
#include "gc/Utils/Error.h"
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#endif

namespace ov {
namespace mlir {

using ::mlir::OwningOpRef;
using ::mlir::ModuleOp;
using ::mlir::ExecutionEngine;
using ::mlir::ModuleOp;

enum MlirMode {
    MLIR_MODE_TPP,
    MLIR_MODE_GC,
    MLIR_MODE_GC_GPU,
    MLIR_MODE_DEFAULT,
};

class MLIROp;

class MLIREvaluateBase {
public:
    static std::shared_ptr<MLIREvaluateBase> create(OwningOpRef<ModuleOp> module, MlirMode mode);

    virtual bool requires_packed_args() const = 0;
    virtual bool invoke(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) = 0;
    virtual bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) = 0;
    virtual ~MLIREvaluateBase() = default;
};

#if defined(GRAPH_COMPILER) && defined(GC_ENABLE_IMEX)

class MLIREvaluateGcGPU : public MLIREvaluateBase {
    gc::gpu::OclModuleBuilder module;
    // TODO: keep compiled oclModule instead of OclModuleBuilder
    // std::shared_ptr<mlir::gc::gpu::OclModule> oclModule;
public:
    MLIREvaluateGcGPU(OwningOpRef<ModuleOp> _module/*, OclRuntime / cl_device_id + context / cl_command_queue */);

    bool requires_packed_args() const override { /*return !oclModule->isStatic;*/ return false; }
    bool invoke(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override;
    bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override;
};

#endif // GRAPH_COMPILER && GC_ENABLE_IMEX

class MLIREvaluate : public MLIREvaluateBase {
    OwningOpRef<ModuleOp> module;  // FIXME: needs to be kept?
    std::unique_ptr<ExecutionEngine> engine;

public:

    MLIREvaluate(OwningOpRef<ModuleOp> _module, MlirMode mode);
    bool requires_packed_args() const override { return true; }
    bool invoke(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override { return false; };
    bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override;
};


// Maps [output index][dimension index] -> [input index][dimension index] to infer shapes for entire subgraph
using DimensionsMap = std::vector<std::vector<std::tuple<size_t, size_t>>>;


class OPENVINO_API MLIROp : public ov::op::Op {
    std::shared_ptr<MLIREvaluateBase> engine;
    OVOutputTypes output_types;
    DimensionsMap dimensions_map;

public:

    OPENVINO_OP("MLIROp");

    MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluateBase> engine,
                                        const OVOutputTypes& output_types, const DimensionsMap& dimensions_map);
    void validate_and_infer_types() override;
    NodePtr clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override;
    bool has_evaluate() const override;
};

} // namespace mlir
} // namespace ov