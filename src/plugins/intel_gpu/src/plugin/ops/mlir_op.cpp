// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl/ocl_device_detector.hpp"
#include "ocl/ocl_common.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/core/partial_shape.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/generic_primitive.hpp"
#include "ocl/ocl_memory.hpp"

#include <iostream>
#include <fstream>

#if defined(GRAPH_COMPILER) && defined(GC_ENABLE_IMEX)
#include "runtime/ocl/ocl_stream.hpp"
#include "gc/ExecutionEngine/OpenCLRuntime/OpenCLRuntimeWrappers.h"
#endif

namespace ov {
namespace op {
namespace mlir {
using MLIRSubgraph = ov::op::Op;
}  // namespace mlir
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

void CreateMLIRSubgraphOp(ProgramBuilder& p, const std::shared_ptr<ov::op::mlir::MLIRSubgraph>& op) {
    cldnn::generic_primitive::execute_function execute_f = [op](
            const std::vector<cldnn::event::ptr>& dependent_events,
            cldnn::stream& stream,
            const std::vector<cldnn::memory::ptr>& inputs,
            const std::vector<cldnn::memory::ptr>& outputs) {
        // Synchronization as evalute() may be a CPU code
        if (stream.get_queue_type() == cldnn::QueueTypes::out_of_order) {
            for (auto& ev : dependent_events) {
                ev->wait();
            }
        } else {
            stream.finish();
        }

        cldnn::event::ptr ev = stream.create_user_event(false);

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;
        // to keep cl_mem objects alive
        std::vector<cl_mem> cl_mem_refs;
        cl_mem_refs.reserve(inputs.size() + outputs.size());
        std::vector<bool> is_usm_ptr;

        auto process_buffer = [&stream, &cl_mem_refs, &is_usm_ptr](cldnn::memory::ptr mem, ov::TensorVector& tensors) {
            switch (mem->get_allocation_type()) {
                case cldnn::allocation_type::cl_mem: {
                    auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_buffer*>(mem.get());
                    cl_mem cl_buff = gpu_buff->get_buffer().get();

                    // Keep the cl_mem reference alive
                    cl_mem_refs.push_back(cl_buff);
                    std::cout << "cl buff: " << &cl_mem_refs.back() << std::endl;
                    tensors.push_back(make_tensor(mem->get_layout(), &cl_mem_refs.back()));
                    is_usm_ptr.push_back(false);
                    break;
                }
                case cldnn::allocation_type::usm_host:
                case cldnn::allocation_type::usm_shared:
                case cldnn::allocation_type::usm_device: {
                    auto usm_ptr = mem->buffer_ptr();
                    std::cout << "usm ptr: " << usm_ptr << std::endl;
                    // Seems to only occur with OOO queues sometimes. Can't reproduce this anymore, uncomment if needed.
                    // HACK: force move to device, can we do better than this?
                    // auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_usm*>(mem.get());
                    // auto& usm_helper = gpu_buff->get_buffer().getUsmHelper();
                    // usm_helper.enqueue_memcpy(
                    //     dynamic_cast<cldnn::ocl::ocl_stream&>(stream).get_cl_queue(),
                    //     usm_ptr,
                    //     usm_ptr,
                    //     mem->get_layout().bytes_count());
                    tensors.push_back(make_tensor(mem->get_layout(), usm_ptr));
                    is_usm_ptr.push_back(true);
                    break;
                }
                default:
                    OPENVINO_THROW("Unsupported memory type");
            }
        };

        for (size_t i = 0; i < inputs.size(); i++) {
            process_buffer(inputs[i], input_host_tensors);
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            process_buffer(outputs[i], output_host_tensors);
        }

        ov::EvaluationContext meta;
        cl_command_queue queue = nullptr;
        if (auto ocl_stream = dynamic_cast<cldnn::ocl::ocl_stream*>(&stream)) {
            queue = ocl_stream->get_cl_queue().get();
            meta["queue"] = queue;
        }
        meta["is_usm_ptr_vector"] = std::cref(is_usm_ptr);

        OPENVINO_ASSERT(op->evaluate(
                        output_host_tensors, input_host_tensors, meta),
                        "[GPU] Couldn't execute MLIROp ", op->get_friendly_name());

        ev->set();
        return ev;
    };
    cldnn::generic_primitive::shape_infer_function shape_infer_f = [&op](
            const std::vector<ov::PartialShape>& input_shapes) -> std::vector<ov::PartialShape> {
        // Dummy shape infer
        return {input_shapes[0]};
    };

    auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);
    const size_t num_outputs = op->get_output_size();

    const cldnn::generic_primitive primitive(layerName,
                                             inputs,
                                             execute_f,
                                             shape_infer_f,
                                             num_outputs,
                                             get_output_data_types(op));

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(mlir, MLIRSubgraph);

}  // namespace intel_gpu
}  // namespace ov
