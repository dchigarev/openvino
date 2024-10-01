// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/core/partial_shape.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/generic_primitive.hpp"
#include "ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_base_event.hpp"

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

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
        cldnn::event::ptr ev = stream.create_user_event(false);

        ov::TensorVector input_gpu_tensors;
        ov::TensorVector output_gpu_tensors;
        std::vector<bool> is_usm_ptr;
        input_gpu_tensors.reserve(inputs.size());
        output_gpu_tensors.reserve(outputs.size());
        is_usm_ptr.reserve(inputs.size() + outputs.size());

        auto process_buffer = [&stream, &is_usm_ptr](cldnn::memory::ptr mem, ov::TensorVector& tensors) {
            switch (mem->get_allocation_type()) {
                case cldnn::allocation_type::cl_mem: {
                    auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_buffer*>(mem.get());
                    cl_mem cl_buff = gpu_buff->get_buffer().get();

                    tensors.push_back(make_tensor(mem->get_layout(), static_cast<void*>(cl_buff)));
                    is_usm_ptr.push_back(false);
                    break;
                }
                case cldnn::allocation_type::usm_host:
                case cldnn::allocation_type::usm_shared:
                case cldnn::allocation_type::usm_device: {
                    auto usm_ptr = mem->buffer_ptr();
                    // Seems to only occur with Out-Of-Order queues sometimes. Can't reproduce this anymore, uncomment if needed.
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
            process_buffer(inputs[i], input_gpu_tensors);
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            process_buffer(outputs[i], output_gpu_tensors);
        }

        ov::EvaluationContext meta;
        if (auto ocl_stream = dynamic_cast<cldnn::ocl::ocl_stream*>(&stream)) {
            cl_command_queue queue = ocl_stream->get_cl_queue().get();
            meta.insert(ov::intel_gpu::ocl_queue(queue));
        }
        meta.insert(ov::intel_gpu::memory_type::is_kernel_arg_usm(is_usm_ptr));

        std::vector<ov::intel_gpu::gpu_handle_param> wait_list;
        if (stream.get_queue_type() == cldnn::QueueTypes::out_of_order) {
            wait_list.reserve(dependent_events.size());
            for (auto& ev : dependent_events) {
                if (auto ocl_ev = dynamic_cast<cldnn::ocl::ocl_base_event*>(ev.get())) {
                    wait_list.push_back(ocl_ev);
                } else {
                    // TODO: maybe we should simply wait this event instead of throwing an error?
                    // ev->wait();
                    OPENVINO_THROW("Unsupported event type");
                }
            }
        }
        meta.insert(ov::intel_gpu::wait_list(wait_list));

        OPENVINO_ASSERT(op->evaluate(
                        output_gpu_tensors, input_gpu_tensors, meta),
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
