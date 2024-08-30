// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/extension.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp>
#include "opencl_helper_instance.hpp"

using testing::ElementsAreArray;

static std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>(ov::test::utils::getExecutableDirectory(),
                                                    std::string("openvino_template_extension") + OV_BUILD_POSTFIX);
}

static std::string model_full_path(const char* path) {
    return ov::util::make_path<char>("/home/jovyan/openvino/src/tests/functional/plugin/shared/models",
                                     path);
}

static void infer_model(ov::Core& core,
                        ov::CompiledModel& model,
                        std::vector<float>& input_values,
                        const std::vector<float>& expected) {
    auto input_tensor = ov::Tensor(ov::element::f32, model.input(0).get_shape(), input_values.data());

    auto infer_req = model.create_infer_request();
    infer_req.set_input_tensor(input_tensor);
    infer_req.infer();

    auto computed = infer_req.get_output_tensor(0);
    EXPECT_THAT(expected, ElementsAreArray(computed.data<const float>(), computed.get_size()));
}

struct OpenCL2 {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;
    bool _supports_usm;

    explicit OpenCL2(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
        // get Intel GPU OCL device, create context and queue
        {
            std::vector<cl::Device> devices;
            std::vector<cl::Platform> platforms;
            const unsigned int refVendorID = 0x8086;

            cl::Platform::get(&platforms);
            for (auto& p : platforms) {
                p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                        _device = d;
                        _context = cl::Context(_device);
                        break;
                    }
                }
            }

            cl_command_queue_properties props = 0;// CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);
            set_usm_available();
        }
    }

    explicit OpenCL2(cl_context context) {
        // user-supplied context handle
        _context = cl::Context(context, true);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);

        auto extensions = _device.getInfo<CL_DEVICE_EXTENSIONS>();
        _supports_usm = extensions.find("cl_intel_unified_shared_memory") != std::string::npos;

        cl_command_queue_properties props = 0;// CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
        set_usm_available();
    }

    clHostMemAllocINTEL_fn _host_mem_alloc_fn = nullptr;
    clMemFreeINTEL_fn _mem_free_fn = nullptr;
    clSharedMemAllocINTEL_fn _shared_mem_alloc_fn = nullptr;
    clDeviceMemAllocINTEL_fn _device_mem_alloc_fn = nullptr;
    clSetKernelArgMemPointerINTEL_fn _set_kernel_arg_mem_pointer_fn = nullptr;
    clEnqueueMemcpyINTEL_fn _enqueue_memcpy_fn = nullptr;
    clEnqueueMemFillINTEL_fn _enqueue_mem_fill_fn = nullptr;
    clEnqueueMemsetINTEL_fn _enqueue_memset_fn = nullptr;
    clGetMemAllocInfoINTEL_fn _get_mem_alloc_info_fn = nullptr;

private:
    void set_usm_available() {
        auto extensions = _device.getInfo<CL_DEVICE_EXTENSIONS>();
        _supports_usm = extensions.find("cl_intel_unified_shared_memory") != std::string::npos;
        load_ext_funcs();
    }

    void load_ext_funcs() {
        if (!_supports_usm) {
            return;
        }

        _host_mem_alloc_fn = queryCLExtFunc<clHostMemAllocINTEL_fn>(_device.get(), "clHostMemAllocINTEL");
        _mem_free_fn = queryCLExtFunc<clMemFreeINTEL_fn>(_device.get(), "clMemFreeINTEL");
        _shared_mem_alloc_fn = queryCLExtFunc<clSharedMemAllocINTEL_fn>(_device.get(), "clSharedMemAllocINTEL");
        _device_mem_alloc_fn = queryCLExtFunc<clDeviceMemAllocINTEL_fn>(_device.get(), "clDeviceMemAllocINTEL");
        _set_kernel_arg_mem_pointer_fn = queryCLExtFunc<clSetKernelArgMemPointerINTEL_fn>(_device.get(), "clSetKernelArgMemPointerINTEL");
        _enqueue_memcpy_fn = queryCLExtFunc<clEnqueueMemcpyINTEL_fn>(_device.get(), "clEnqueueMemcpyINTEL");
        _enqueue_mem_fill_fn = queryCLExtFunc<clEnqueueMemFillINTEL_fn>(_device.get(), "clEnqueueMemFillINTEL");
        _enqueue_memset_fn = queryCLExtFunc<clEnqueueMemsetINTEL_fn>(_device.get(), "clEnqueueMemsetINTEL");
        _get_mem_alloc_info_fn = queryCLExtFunc<clGetMemAllocInfoINTEL_fn>(_device.get(), "clGetMemAllocInfoINTEL");
    }

    template<typename T>
    T queryCLExtFunc(cl_device_id dev, const char *FuncName) {
        cl_platform_id CurPlatform;
        clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                                    &CurPlatform, nullptr);
        void *ret = clGetExtensionFunctionAddressForPlatform(CurPlatform, FuncName);

        if (!ret) {
            fflush(stderr);
            abort();
        }
        return reinterpret_cast<T>(ret);
    }
};

template<typename T>
static ov::Tensor allocate_usm_tensor(size_t byte_size, ov::CompiledModel& compiledModel, ov::element::Type type, std::vector<T> &input_values) {
    auto context = compiledModel.get_context();
    // context.create_tensor(ov::element::f32, {1}, clBuffer[0]);

    auto& oclContext = static_cast<ov::intel_gpu::ocl::ClContext&>(context);
    auto oclInstance = std::make_shared<OpenCL2>(oclContext.get());
    cl_int err;
    // void* usm_ptr = oclInstance->_usm_helper->allocate_host(
    //     /*properties=*/nullptr,
    //     /*size=*/byte_size,
    //     /*alignment=*/0,
    //     /*err_code_return=*/&err);

    // std::cout << "STATUS: " << err << std::endl;

    // err = oclInstance->_usm_helper->enqueue_memcpy(
    //     oclInstance->_queue,
    //     /*dst_ptr=*/usm_ptr,
    //     /*src_ptr=*/input_values.data(),
    //     byte_size,
    //     /*blocking=*/true,
    //     /*wait_list=*/nullptr,
    //     /*ret_event=*/nullptr);

    // auto sz = oclInstance->_usm_helper->get_usm_allocation_size(usm_ptr);

    void* usm_ptr = oclInstance->_host_mem_alloc_fn(
        oclInstance->_context.get(),
        // oclContext.get(),
        // oclInstance->_device.get(),
        nullptr, // properties,
        byte_size,// size,
        0, // alignment,
        &err);// err_code_ret);
    std::cout << "allocated: " << usm_ptr << std::endl;

        // auto ev = stream.create_base_event();
    // cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
    // enqueueFillUsm call will never finish. Driver bug? Uncomment when fixed. Some older drivers doesn't support enqueueFillUsm call at all.
    // cl_stream.get_usm_helper().enqueue_fill_mem<unsigned char>(cl_stream.get_cl_queue(), _buffer.get(), pattern, _bytes_count, nullptr, &ev_ocl)
    // Workarounded with enqeue_memcopy. ToDo: Remove below code. Uncomment above.

    std::vector<float> temp_buffer(byte_size, 0.5);
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
        usm_ptr, // dst_ptr
        temp_buffer.data(), // src_ptr
        byte_size,
        0, // wait_list_size == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
        nullptr, // wait_list == nullptr ? nullptr : reinterpret_cast<const cl_event*>(&wait_list->front()),
        &tmp);// ret_event == nullptr ? nullptr : &tmp);

    std::cout << "sz " << byte_size << std::endl;
    std::cout << "STATUS2: " << err << " | PTR: " << usm_ptr << std::endl;
    ov::Shape shp{{64, 128}};
    auto tensor = oclContext.create_tensor(type, shp, usm_ptr);
    return tensor;
    // auto usmTens = oclContext.create_usm_device_tensor(ov::element::f32, {1});
    // usmTens.
    // auto oclInstance = std::make_shared<OpenCL>(oclContext.get());
    // return ov::Tensor(type, {size}, ov::runtime::AllocationType::usm_shared);
}

TEST(MLIRExecution, SimpleMatmul) {
    std::vector<float> input_values(64 * 128, 0.5f);

    ov::Core core;
    core.add_extension(get_extension_path());
    auto ov_model = core.read_model(
        "/home/jovyan/openvino/src/plugins/intel_gpu/tests/functional/mlir_op/models/matmul_64_128_f32.xml");
    auto compiled_model = core.compile_model(ov_model, "GPU");

    auto infer_req = compiled_model.create_infer_request();
    auto tensor = allocate_usm_tensor(64 * 128 * 4, compiled_model, ov::element::f32, input_values);
    auto tens = infer_req.get_output_tensor(0);
    auto shp = tens.get_shape();
    std::cout << "SHP: ";
    for (auto i : shp) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    infer_req.set_input_tensor(tensor);
    infer_req.infer();
    // infer_req.wait();
    // auto computed = infer_req.get_output_tensor(0);
    // infer_model(core, compiled_model, input_values, expected);
}

TEST(Extension, smoke_XmlModelWithExtensionFromDSO) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="2,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="operation" id="1" type="Identity" version="extension">
            <data  add="11"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{1, 2, 3, 4, 5, 6, 7, 8};

    ov::Core core;
    core.set_property("CPU", {{ov::hint::inference_precision.name(), ov::element::f32.get_type_name()}});
    core.add_extension(get_extension_path());
    auto weights = ov::Tensor();
    auto ov_model = core.read_model(model, weights);
    auto compiled_model = core.compile_model(ov_model);

    infer_model(core, compiled_model, input_values, expected);
}
