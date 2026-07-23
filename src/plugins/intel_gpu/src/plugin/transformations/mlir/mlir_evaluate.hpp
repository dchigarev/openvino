// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::mlir {

using ::mlir::ExecutionEngine;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;

class MLIREvaluateBase {
public:
    virtual bool requires_packed_args() const = 0;
    virtual bool invoke(const ov::TensorVector& inputs,
                        ov::TensorVector& outputs,
                        const ov::EvaluationContext& evaluationContext) = 0;
    virtual bool invoke_packed(std::vector<void*>& args,
                               const ov::EvaluationContext& evaluationContext) = 0;
    virtual ~MLIREvaluateBase() = default;
};

class MLIREvaluateGcGPU : public MLIREvaluateBase {
    std::shared_ptr<const ::mlir::gc::gpu::OclModule> module;

public:
    MLIREvaluateGcGPU(OwningOpRef<ModuleOp> _module,
                      std::shared_ptr<ov::EvaluationContext> loweringContext);

    bool requires_packed_args() const override { return !module->isStatic; }
    bool invoke(const ov::TensorVector& inputs,
                ov::TensorVector& outputs,
                const ov::EvaluationContext& evaluationContext) override;
    bool invoke_packed(std::vector<void*>& args,
                       const ov::EvaluationContext& evaluationContext) override;

private:
    ::mlir::gc::gpu::OclContext build_ocl_context(const ov::EvaluationContext& evaluationContext,
                                                  std::vector<void*>& waitList);
    static void maybe_set_result_events(const ov::EvaluationContext& evaluationContext,
                                        ::mlir::gc::gpu::OclContext& ctx);
};

}  // namespace ov::mlir
