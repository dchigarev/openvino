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

namespace ov {
namespace op {
namespace mlir {
using MLIRSubgraph = ov::op::Op;
}  // namespace mlir
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

void writeArrayToFile(float* array, size_t sz, const std::string& filename) {
    // Open the file in text mode
    std::ofstream file(filename);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write each element to the file in a human-readable format
    for (size_t i = 0; i < sz; ++i) {
        file << array[i];
        if (i < sz - 1) {
            file << " "; // Separate values with a space
        }
    }

    if (!file) {
        std::cerr << "Error: Write to file " << filename << " failed." << std::endl;
    } else {
        std::cout << "Array written to file " << filename << " successfully." << std::endl;
    }

    // Close the file
    file.close();
}

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
        std::vector<float> arr(128 * 128, 0.5f);

        // auto process_memory = [&stream](cldnn::memory::ptr mem, ov::TensorVector& tensors) {
        //     switch (mem->get_allocation_type()) {
        //         case cldnn::allocation_type::cl_mem: {
        //             auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_buffer*>(mem.get());
        //             cl_mem cl_buff = gpu_buff->get_buffer().get();
        //             tensors.push_back(make_tensor(mem->get_layout(), &cl_buff));
        //             std::cout << " cl_mem: " << &cl_buff;

        //             // auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_buffer*>(mem.get());
        //             // cl_mem cl_buff = gpu_buff->get_buffer().get();
        //             // std::cout << " cl_buff: " << &cl_buff;
        //             // tensors.push_back(make_tensor(mem->get_layout(), &cl_buff));
        //             break;
        //         }
        //         case cldnn::allocation_type::usm_host:
        //         case cldnn::allocation_type::usm_shared:
        //         case cldnn::allocation_type::usm_device: {
        //             auto usm_ptr = mem->buffer_ptr();
        //             auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_usm*>(mem.get());
        //             auto& usm_helper = gpu_buff->get_buffer().getUsmHelper();
        //             // HACK: force move to device, can we do better than this?
        //             usm_helper.enqueue_memcpy(
        //                 dynamic_cast<cldnn::ocl::ocl_stream&>(stream).get_cl_queue(),
        //                 usm_ptr,
        //                 usm_ptr,
        //                 mem->get_layout().bytes_count());
        //             std::cout << " usm_ptr: " << usm_ptr;
        //             tensors.push_back(make_tensor(mem->get_layout(), usm_ptr));
        //             break;
        //         }
        //         default:
        //             throw std::runtime_error("Unsupported memory type");
        //     }
        // };

        // for (size_t i = 0; i < inputs.size(); i++) {
        //     std::cout << "Input " << i;
        //     process_memory(inputs[i], input_host_tensors);
        //     std::cout << std::endl;
        // }

        // for (size_t i = 0; i < outputs.size(); i++) {
        //     std::cout << "Output " << i;
        //     process_memory(outputs[i], output_host_tensors);
        //     std::cout << std::endl;
        // }

        for (size_t i = 0; i < inputs.size(); i++) {
            auto usm_ptr = inputs[i]->buffer_ptr();
            if (usm_ptr == nullptr) {
                auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_buffer*>(inputs[i].get());
                cl_mem cl_buff = gpu_buff->get_buffer().get();
                input_host_tensors.push_back(make_tensor(inputs[i]->get_layout(), cl_buff));
                std::cout << "input " << i << " is cl_mem: " << cl_buff << std::endl;
            } else {
                // copy data to usm_ptr
                auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_usm*>(inputs[i].get());
                auto& usm_helper = gpu_buff->get_buffer().getUsmHelper();
                usm_helper.enqueue_memcpy(
                    dynamic_cast<cldnn::ocl::ocl_stream&>(stream).get_cl_queue(),
                    usm_ptr,
                    usm_ptr,
                    inputs[i]->get_layout().bytes_count());
                input_host_tensors.push_back(make_tensor(inputs[i]->get_layout(), usm_ptr));
                std::cout << "input " << i << " is USM: " << usm_ptr << " sz: " << inputs[i]->get_layout().bytes_count() << std::endl;
            }
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            auto usm_ptr = outputs[i]->buffer_ptr();
            if (usm_ptr == nullptr) {
                auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_buffer*>(outputs[i].get());
                cl_mem cl_buff = gpu_buff->get_buffer().get();
                output_host_tensors.push_back(make_tensor(outputs[i]->get_layout(), &cl_buff));
                std::cout << "output " << i << " is cl_mem: " << &cl_buff << std::endl;
            } else {
                output_host_tensors.push_back(make_tensor(outputs[i]->get_layout(), usm_ptr));
                std::cout << "output " << i << " is USM: " << usm_ptr << std::endl;
            }
        }

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute MLIROp ", op->get_friendly_name());

        // for testing purposes
        // float* output = reinterpret_cast<float*>(outputs[0]->lock(stream, cldnn::mem_lock_type::read));
        // writeArrayToFile(output, 64 * 128, "/home/jovyan/openvino/out_after.txt");
        // outputs[0]->unlock(stream);

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
