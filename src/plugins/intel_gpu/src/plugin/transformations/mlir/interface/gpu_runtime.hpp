// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "../common/typedefs.hpp"
#include "intel_gpu/runtime/event.hpp"

namespace cldnn {
class stream;
class engine;
struct network;
class primitive_inst;
}  // namespace cldnn

namespace ov {
class Node;
}  // namespace ov

namespace ov::intel_gpu::mlir {

struct MLIRGpuRuntime {
public:
    virtual ~MLIRGpuRuntime() = default;
    MLIRGpuRuntime(const MLIRGpuRuntime&) = delete;
    MLIRGpuRuntime& operator=(const MLIRGpuRuntime&) = delete;

protected:
    MLIRGpuRuntime() = default;
    friend struct ::cldnn::network;
    inline static std::unique_ptr<MLIRGpuRuntime> (*create)(cldnn::stream&, cldnn::engine&) = nullptr;
};

class MLIRGpuProgram {
public:
    virtual ~MLIRGpuProgram() = default;
    MLIRGpuProgram(const MLIRGpuProgram&) = delete;
    MLIRGpuProgram& operator=(const MLIRGpuProgram&) = delete;

    virtual void wait_compiled() = 0;
    virtual cldnn::event::ptr execute(MLIRGpuRuntime& runtime,
                                      const ov::Node& op,
                                      cldnn::primitive_inst& instance,
                                      const std::vector<cldnn::event::ptr>& deps,
                                      bool need_event) = 0;

    // Physical orders of the inputs, baked into the memref argument types of the compiled function.
    const InputOrders& get_input_orders() const {
        return input_orders;
    }

protected:
    explicit MLIRGpuProgram(InputOrders input_orders) : input_orders(std::move(input_orders)) {}

private:
    InputOrders input_orders;
};

void register_mlir_gpu_runtime();

}  // namespace ov::intel_gpu::mlir
