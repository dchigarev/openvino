// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_tensors_filling.hpp"

#include <memory>
#include <random>
#include <samples/slog.hpp>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#ifdef HAVE_DEVICE_MEM_SUPPORT
#    include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#    include <openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp>
#endif

#include <iostream>

namespace gpu {

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

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fill_buffer_random(void* inputBuffer,
                        size_t elementsNum,
                        T rand_min = std::numeric_limits<uint8_t>::min(),
                        T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    uniformDistribution<T2> distribution(rand_min, rand_max);
    auto inputBufferData = static_cast<T*>(inputBuffer);
    std::cout << "filling with 0.5" << std::endl;
    for (size_t i = 0; i < elementsNum; i++) {
        // inputBufferData[i] = static_cast<T>(distribution(gen));
        inputBufferData[i] = static_cast<T>(0.5);
    }
}

void fill_buffer(void* inputBuffer, size_t elementsNum, const ov::element::Type& type) {
    if (type == ov::element::f32) {
        fill_buffer_random<float, float>(inputBuffer, elementsNum);
    } else if (type == ov::element::f64) {
        fill_buffer_random<double, double>(inputBuffer, elementsNum);
    } else if (type == ov::element::f16) {
        fill_buffer_random<ov::float16, float>(inputBuffer, elementsNum);
    } else if (type == ov::element::i32) {
        fill_buffer_random<int32_t, int32_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::i64) {
        fill_buffer_random<int64_t, int64_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::u8) {
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        fill_buffer_random<uint8_t, uint32_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::i8) {
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        fill_buffer_random<int8_t, int32_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::u16) {
        fill_buffer_random<uint16_t, uint16_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::i16) {
        fill_buffer_random<int16_t, int16_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::boolean) {
        fill_buffer_random<uint8_t, uint32_t>(inputBuffer, elementsNum, 0, 1);
    } else {
        OPENVINO_THROW("Requested type is not supported");
    }
}

std::map<std::string, ov::TensorVector> get_remote_input_tensors(
    const std::map<std::string, std::vector<std::string>>& inputFiles,
    const std::vector<benchmark_app::InputsInfo>& app_inputs_info,
    const ov::CompiledModel& compiledModel,
    std::vector<BufferType>& clBuffer,
    size_t num_requests) {
#ifdef HAVE_DEVICE_MEM_SUPPORT
    slog::info << "Device memory will be used for input and output blobs" << slog::endl;
    if (inputFiles.size()) {
        slog::warn << "Device memory supports only random data at this moment, input images will be ignored"
                   << slog::endl;
    }

    std::map<std::string, ov::TensorVector> remoteTensors;
    auto context = compiledModel.get_context();
    // context.create_tensor(ov::element::f32, {1}, clBuffer[0]);

    auto& oclContext = static_cast<ov::intel_gpu::ocl::ClContext&>(context);
    // auto usmTens = oclContext.create_usm_device_tensor(ov::element::f32, {1});
    // usmTens.
    auto oclInstance = std::make_shared<gpu::OpenCL>(oclContext.get());
    // _shared_mem_alloc_fn(_ctx.get(), _device.get(), properties, size, alignment, err_code_ret);
    // auto oclInstance = std::make_shared<gpu::OpenCL>();

    // auto& cl_stream = downcast<ocl_stream>(stream);

    std::vector<cl::Device> devices = oclInstance->_context.getInfo<CL_CONTEXT_DEVICES>();

    // Print the details of each device
    for (size_t i = 0; i < devices.size(); i++) {
        std::string deviceName = devices[i].getInfo<CL_DEVICE_NAME>();
        std::string deviceVendor = devices[i].getInfo<CL_DEVICE_VENDOR>();
        cl_device_type deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();

        std::cout << "Device " << i << ": " << deviceName << std::endl;
        std::cout << "  Vendor: " << deviceVendor << std::endl;
        std::cout << "  Type: ";
        if (deviceType & CL_DEVICE_TYPE_CPU) {
            std::cout << "CPU ";
        }
        if (deviceType & CL_DEVICE_TYPE_GPU) {
            std::cout << "GPU ";
        }
        if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
            std::cout << "Accelerator ";
        }
        if (deviceType & CL_DEVICE_TYPE_DEFAULT) {
            std::cout << "Default ";
        }
        std::cout << std::endl;
    }


    bool done = false;

    for (size_t i = 0; i < num_requests; i++) {
        for (auto& inputs_info : app_inputs_info) {
            for (auto& input : inputs_info) {
                // Fill random
                slog::info << "Prepare remote blob for input '" << input.first << "' with random values ("
                           << std::string((input.second.is_image() ? "image" : "some binary data")) << " is expected)"
                           << slog::endl;

                // Creating and filling shared buffers
                cl_int err;
                auto elementsNum = std::accumulate(begin(input.second.dataShape),
                                                   end(input.second.dataShape),
                                                   1,
                                                   std::multiplies<size_t>());
                auto inputSize = elementsNum * input.second.type.bitwidth() / 8;
                std::cout << "Element num: " << elementsNum << " | " << input.second.type.to_string()  << " | inp_sz: " << inputSize << std::endl;
                size_t _bytes_count = inputSize * 4;

                void* usm_p = oclInstance->_host_mem_alloc_fn(
                    oclInstance->_context.get(),
                    // oclContext.get(),
                    // oclInstance->_device.get(),
                    nullptr, // properties,
                    _bytes_count,// size,
                    0, // alignment,
                    &err// err_code_ret
                );
                std::cout << "allocated: " << usm_p << std::endl;

                    // auto ev = stream.create_base_event();
                // cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
                // enqueueFillUsm call will never finish. Driver bug? Uncomment when fixed. Some older drivers doesn't support enqueueFillUsm call at all.
                // cl_stream.get_usm_helper().enqueue_fill_mem<unsigned char>(cl_stream.get_cl_queue(), _buffer.get(), pattern, _bytes_count, nullptr, &ev_ocl)
                // Workarounded with enqeue_memcopy. ToDo: Remove below code. Uncomment above.

                std::vector<float> temp_buffer(_bytes_count, 0.5);
                // TODO: Do we really need blocking call here? Non-blocking one causes accuracy issues right now, but hopefully it can be fixed in more performant way.
                // const bool blocking = true;
                    
                // oclInstance->_enqueue_memcpy_fn(
                //     oclInstance->_queue.get(),// cl_stream.get_cl_queue(),
                //     usm_p,// _buffer.get(),
                //     temp_buffer.data(),
                //     _bytes_count,
                //     blocking,
                //     nullptr,
                //     &ev_ocl);


                cl_event tmp;
                err = oclInstance->_enqueue_memcpy_fn(
                    oclInstance->_queue.get(),
                    static_cast<cl_bool>(true), // blocking
                    usm_p, // dst_ptr
                    temp_buffer.data(), // src_ptr
                    _bytes_count,
                    0, // wait_list_size == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
                    nullptr, // wait_list == nullptr ? nullptr : reinterpret_cast<const cl_event*>(&wait_list->front()),
                    &tmp// ret_event == nullptr ? nullptr : &tmp);
                );
                // clBuffer.push_back(
                //     cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err));
                // clBuffer.push_back(
                //     cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize));

                // cl::Event unmapEvent;
                // void* mappedPtr = oclInstance->_queue.enqueueMapBuffer(clBuffer.back(),
                //                                                        CL_TRUE,
                //                                                        CL_MAP_WRITE,
                //                                                        // CL_MEM_READ_WRITE,
                //                                                        0,
                //                                                        (cl::size_type)inputSize);
                // void* mappedPtr = usm_p;
                // err = oclInstance->_queue.finish();
                // auto tensor =
                //     oclContext.create_tensor(input.second.type, input.second.dataShape, clBuffer.back().get());
                auto tensor =
                    oclContext.create_tensor(input.second.type, input.second.dataShape, usm_p);
                remoteTensors[input.first].push_back(tensor);
                // fill_buffer(mappedPtr, elementsNum, input.second.type);
                // if (!done)
                //     writeArrayToFile((float*)mappedPtr, inputSize, "__inp1b.txt");

                // err = oclInstance->_queue.enqueueUnmapMemObject(clBuffer.back(), mappedPtr);
                // queue.enqueueUnmapMemObject(gpuBuffer, mappedPtr);
                // oclInstance->_queue.finish();
                // unmapEvent.wait();
                std::cout << "unmap ERR: " << err << std::endl;
                // err = oclInstance->_queue.finish();
                std::cout << "finish ERR: " << err << std::endl;
                // if (!done) {
                //     mappedPtr = oclInstance->_queue.enqueueMapBuffer(clBuffer.back(),
                //                                             CL_TRUE,
                //                                             CL_MEM_READ_WRITE,
                //                                             0,
                //                                             (cl::size_type)inputSize, NULL, NULL, &err);
                //     std::cout << "map ERR: " << err << std::endl;
                //     writeArrayToFile((float*)mappedPtr, inputSize, "__inp1a.txt");
                //     oclInstance->_queue.enqueueUnmapMemObject(clBuffer.back(), mappedPtr);
                // }

                done = true;
            }
        }
    }
    return remoteTensors;
#else
    OPENVINO_THROW("Device memory requested for GPU device, but OpenCL was not linked");
#endif
}

ov::Shape get_static_shape(const ov::Output<const ov::Node>& compiled_output) {
    // FIXME: this is a WA for case when original model has internal dynamism (NonMaxSuppression)
    // and runtime has static output due to conversions to legacy op and lack of dynamism support
    // to be removed along with dynamism support
    const auto& compiled_pshape = compiled_output.get_partial_shape();
    if (compiled_pshape.is_static())
        return compiled_pshape.to_shape();
    else if (compiled_pshape.rank().is_dynamic())
        OPENVINO_THROW("Benchmark App - NOT IMPLEMENTED - Output of dynamic rank is not supported for remote tensor. ",
                       "Output: ",
                       compiled_output);
    ov::Shape shape;
    for (const auto& dimension : compiled_pshape) {
        if (dimension.get_interval().has_upper_bound())
            shape.push_back(static_cast<ov::Shape::value_type>(dimension.get_max_length()));
        else
            OPENVINO_THROW("Benchmark App - NOT IMPLEMENTED - Fully dynamic output dimensions are not supported "
                           "for remote tensor. ",
                           "Output: ",
                           compiled_output);
    }
    return shape;
}

std::map<std::string, ov::Tensor> get_remote_output_tensors(const ov::CompiledModel& compiledModel,
                                                            std::map<std::string, ::gpu::BufferType>& clBuffer) {
#ifdef HAVE_DEVICE_MEM_SUPPORT
    std::map<std::string, ov::Tensor> outputTensors;
    std::shared_ptr<const ov::Model> runtime_model = nullptr;
    for (auto& output : compiledModel.outputs()) {
        auto context = compiledModel.get_context();
        auto& oclContext = static_cast<ov::intel_gpu::ocl::ClContext&>(context);
        auto oclInstance = std::make_shared<OpenCL>(oclContext.get());
        ov::Shape shape = get_static_shape(output);
        cl_int err;
        auto elementsNum = shape_size(shape);
        auto inputSize = elementsNum * output.get_element_type().bitwidth() / 8;

        cl::size_type bufferSize = 0;
        if (clBuffer.find(output.get_any_name()) == clBuffer.end()) {
            clBuffer[output.get_any_name()] =
                cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err);
        } else {
            auto& buff = clBuffer[output.get_any_name()];
            buff.getInfo(CL_MEM_SIZE, &bufferSize);
            if (inputSize != bufferSize) {
                buff = cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err);
            }
        }
        outputTensors[output.get_any_name()] =
            oclContext.create_tensor(output.get_element_type(), shape, clBuffer[output.get_any_name()].get());
    }

    return outputTensors;
#else
    OPENVINO_THROW("Device memory requested for GPU device, but OpenCL was not linked");
#endif
}
}  // namespace gpu
