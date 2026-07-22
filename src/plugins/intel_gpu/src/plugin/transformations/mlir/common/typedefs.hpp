// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/symbol.hpp"


namespace ov {
namespace mlir {

using NodePtr = std::shared_ptr<ov::Node>;
using SymbolPtr = std::shared_ptr<ov::Symbol>;
using InputVector = std::vector<ov::Input<ov::Node>>;
// OVOutputTypes moved to intel_gpu/op/mlir_op.hpp — include that header from
// files that construct MLIROp (convert.cpp does).

} // namespace mlir
} // namespace ov