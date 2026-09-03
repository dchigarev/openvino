// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/matmul.hpp>

#include "../convert_common.hpp"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu::mlir {

// The linalg contraction ops take fixed operand ranks (2 for linalg.matmul, 3 for
// linalg.batch_matmul) and support neither broadcasting nor 1D operands, while OV MatMul
// does all three. Mirrors the cases ConvertMatMul can actually lower.
inline bool isMatMulSupported(const std::shared_ptr<ov::op::v0::MatMul>& node) {
    if (has_dynamic_rank(node)) {
        return false;
    }

    const auto& a = node->get_input_partial_shape(0);
    const auto& b = node->get_input_partial_shape(1);
    if (a.rank().is_dynamic() || b.rank().is_dynamic()) {
        return false;
    }
    // Both transposed at once has no linalg equivalent.
    if (node->get_transpose_a() && node->get_transpose_b()) {
        return false;
    }
    // 1D would need matvec/vecmat/dot.
    if (a.size() < 2 || b.size() < 2) {
        return false;
    }
    // If a.size() > 2, the leading dimensions are folded into M.
    if (!node->get_transpose_a() && b.size() == 2) {
        return true;
    }
    // All leading dimensions are 1: they collapse away, leaving a plain 2D matmul.
    auto unitLeading = [](const PartialShape& s) {
        return std::all_of(s.begin(), s.end() - 2, [](const Dimension& d) {
            return d == 1;
        });
    };
    if (unitLeading(a) && unitLeading(b) && unitLeading(node->get_output_partial_shape(0))) {
        return true;
    }
    // linalg.batch_matmul: exactly one batch dimension, statically equal on both operands.
    return a.size() == 3 && b.size() == 3 && a[0].is_static() && a[0] == b[0];
}

struct ConvertMatMul {
    Operation* operator()(ConversionContext& context, const NodePtr& node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);

        mlir::SmallVector<Value, 2> ins{inputs[0], inputs[1]};

        auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(matmul_node);
        bool isTransposedA = matmul_node->get_transpose_a();
        bool isTransposedB = matmul_node->get_transpose_b();
        assert(!(isTransposedA && isTransposedB));

        auto shapeOf = [](Value tensor) {
            return mlir::cast<RankedTensorType>(tensor.getType()).getShape();
        };
        // [d0, ..., dN-2] x [dN-1]
        auto leadingReassoc = [](int64_t rank) {
            SmallVector<ReassociationIndices> reassoc(1);
            for (int64_t i = 0; i < rank - 1; ++i) {
                reassoc[0].push_back(i);
            }
            reassoc.push_back(ReassociationIndices{rank - 1});
            return reassoc;
        };
        auto unitLeading = [](ArrayRef<int64_t> shape) {  // rank == 2 or all leading dimensions are 1
            auto leading = shape.drop_back(2);
            return std::all_of(leading.begin(), leading.end(), [](int64_t d) {
                return d == 1;
            });
        };
        SmallVector<OpFoldResult> outSizes;
        for (int64_t i = 0, dyn = 0; i < outType.getRank(); ++i) {
            int64_t d = outType.getDimSize(i);
            outSizes.push_back(ShapedType::isDynamic(d) ? OpFoldResult(dynamic_dimensions[dyn++]) : OpFoldResult(builder.getIndexAttr(d)));
        }
        // Fold the leading dimensions of A into M instead of broadcasting B.
        bool foldIntoM = !isTransposedA && shapeOf(ins[1]).size() == 2;
        auto resType = outType;
        auto resDynDims = dynamic_dimensions;
        if (foldIntoM || (unitLeading(shapeOf(ins[0])) && unitLeading(shapeOf(ins[1])) && unitLeading(outType.getShape()))) {
            auto collapse = [&](Value tensor) -> Value {  // Has no-op if rank == 2
                int64_t rank = shapeOf(tensor).size();
                if (rank <= 2) {
                    return tensor;
                }
                return tensor::CollapseShapeOp::create(builder, loc, tensor, leadingReassoc(rank)).getResult();
            };
            ins[0] = collapse(ins[0]);
            ins[1] = collapse(ins[1]);
            int64_t mIdx = isTransposedA ? 1 : 0;
            int64_t m = shapeOf(ins[0])[mIdx];
            int64_t n = outType.getShape().back();
            resType = RankedTensorType::get({m, n}, outType.getElementType());
            resDynDims.clear();
            if (ShapedType::isDynamic(m)) {
                resDynDims.push_back(tensor::DimOp::create(builder, loc, ins[0], mIdx).getResult());
            }
            if (ShapedType::isDynamic(n)) {
                resDynDims.push_back(cast<Value>(outSizes.back()));
            }
        }

        auto empty = tensor::EmptyOp::create(builder, loc, resType, resDynDims);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});
        mlir::SmallVector<Value, 1> outs{fill.getResult(0)};

        Operation* matmul = nullptr;
        if (resType.getRank() > 2) {
            // linalg.batch_matmul takes exactly one batch dimension, equal on both operands.
            assert(resType.getRank() == 3 && shapeOf(ins[0]).size() == 3 && shapeOf(ins[1]).size() == 3);
            assert(shapeOf(ins[0])[0] == shapeOf(ins[1])[0] && !ShapedType::isDynamic(shapeOf(ins[0])[0]));
            if (isTransposedA) {
                matmul = linalg::BatchMatmulTransposeAOp::create(builder, loc, ins, outs);
            } else if (isTransposedB) {
                matmul = linalg::BatchMatmulTransposeBOp::create(builder, loc, ins, outs);
            } else {
                matmul = linalg::BatchMatmulOp::create(builder, loc, ins, outs);
            }
        } else {
            if (isTransposedA) {
                matmul = linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
            } else if (isTransposedB) {
                matmul = linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
            } else {
                matmul = linalg::MatmulOp::create(builder, loc, ins, outs);
            }
            auto result = matmul->getResult(0);
            if (result.getType() != outType) {
                matmul = tensor::ExpandShapeOp::create(builder, loc, outType, result, leadingReassoc(outType.getRank()), outSizes);
            }
        }
        return matmul;
    }
};

}  // namespace ov::intel_gpu::mlir