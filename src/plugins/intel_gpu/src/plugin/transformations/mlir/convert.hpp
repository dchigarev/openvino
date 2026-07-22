// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Plugin-internal header — declares the MLIR transformation entry point.
// Only pulls OV core types, no MLIR / Graph-Compiler headers, so it can be
// included from transformations_pipeline.cpp without transitively dragging
// MLIR into the main plugin compilation.
//
// Only expected to be included from within
//   src/plugins/intel_gpu/src/plugin/
// files, resolved via the src/plugin include path.

#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov::intel_gpu::mlir {

// Runs the OV -> MLIR partitioner + lowering pipeline over the model. Only
// available when the plugin was built with -DENABLE_GRAPH_COMPILER=ON; the
// symbol lives in mlir_lib and does not exist otherwise. Callers must guard
// invocations with `#ifdef GRAPH_COMPILER`.
void transformMLIR(std::shared_ptr<ov::Model> model,
                   std::shared_ptr<ov::EvaluationContext> loweringContext);

}  // namespace ov::intel_gpu::mlir
