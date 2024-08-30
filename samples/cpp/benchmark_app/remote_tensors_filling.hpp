// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(HAVE_GPU_DEVICE_MEM_SUPPORT)
#    define HAVE_DEVICE_MEM_SUPPORT
#    include "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"
#    include "CL/cl_ext.h"
#endif
#include "utils.hpp"

namespace gpu {

#ifdef HAVE_DEVICE_MEM_SUPPORT
using BufferType = cl::Buffer;

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;
    bool _supports_usm;

    explicit OpenCL(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
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

    explicit OpenCL(cl_context context) {
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


// class UsmHelper {
// public:
//     explicit UsmHelper(const cl::Context& ctx, const cl::Device device, bool use_usm) : _ctx(ctx), _device(device) {
//         if (use_usm) {
//             _host_mem_alloc_fn             = try_load_entrypoint<clHostMemAllocINTEL_fn>(_ctx.get(), "clHostMemAllocINTEL");
//             _shared_mem_alloc_fn           = try_load_entrypoint<clSharedMemAllocINTEL_fn>(_ctx.get(), "clSharedMemAllocINTEL");
//             _device_mem_alloc_fn           = try_load_entrypoint<clDeviceMemAllocINTEL_fn>(_ctx.get(), "clDeviceMemAllocINTEL");
//             _mem_free_fn                   = try_load_entrypoint<clMemFreeINTEL_fn>(_ctx.get(), "clMemFreeINTEL");
//             _set_kernel_arg_mem_pointer_fn = try_load_entrypoint<clSetKernelArgMemPointerINTEL_fn>(_ctx.get(), "clSetKernelArgMemPointerINTEL");
//             _enqueue_memcpy_fn             = try_load_entrypoint<clEnqueueMemcpyINTEL_fn>(_ctx.get(), "clEnqueueMemcpyINTEL");
//             _enqueue_mem_fill_fn           = try_load_entrypoint<clEnqueueMemFillINTEL_fn>(_ctx.get(), "clEnqueueMemFillINTEL");
//             _enqueue_memset_fn             = try_load_entrypoint<clEnqueueMemsetINTEL_fn>(_ctx.get(), "clEnqueueMemsetINTEL");
//             _get_mem_alloc_info_fn         = try_load_entrypoint<clGetMemAllocInfoINTEL_fn>(_ctx.get(), "clGetMemAllocInfoINTEL");
//         }
//     }

//     void* allocate_host(const cl_mem_properties_intel *properties, size_t size, cl_uint alignment, cl_int* err_code_ret) const {\
//         if (!_host_mem_alloc_fn)
//             throw std::runtime_error("[CLDNN] clHostMemAllocINTEL is nullptr");
//         return _host_mem_alloc_fn(_ctx.get(), properties, size, alignment, err_code_ret);
//     }

//     void* allocate_shared(const cl_mem_properties_intel *properties, size_t size, cl_uint alignment, cl_int* err_code_ret) const {
//         if (!_shared_mem_alloc_fn)
//             throw std::runtime_error("[CLDNN] clSharedMemAllocINTEL is nullptr");
//         return _shared_mem_alloc_fn(_ctx.get(), _device.get(), properties, size, alignment, err_code_ret);
//     }

//     void* allocate_device(const cl_mem_properties_intel *properties, size_t size, cl_uint alignment, cl_int* err_code_ret) const {
//         if (!_device_mem_alloc_fn)
//             throw std::runtime_error("[CLDNN] clDeviceMemAllocINTEL is nullptr");
//         return _device_mem_alloc_fn(_ctx.get(), _device.get(), properties, size, alignment, err_code_ret);
//     }

//     void free_mem(void* ptr) const {
//         if (!_mem_free_fn)
//             throw std::runtime_error("[CLDNN] clMemFreeINTEL is nullptr");
//         _mem_free_fn(_ctx.get(), ptr);
//     }

//     cl_int set_kernel_arg_mem_pointer(const cl::Kernel& kernel, uint32_t index, const void* ptr) const {
//         if (!_set_kernel_arg_mem_pointer_fn)
//             throw std::runtime_error("[CLDNN] clSetKernelArgMemPointerINTEL is nullptr");
//         return _set_kernel_arg_mem_pointer_fn(kernel.get(), index, ptr);
//     }

//     cl_int enqueue_memcpy(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void *src_ptr,
//                           size_t bytes_count, bool blocking = true, const std::vector<cl::Event>* wait_list = nullptr, cl::Event* ret_event = nullptr) const {
//         if (!_enqueue_memcpy_fn)
//             throw std::runtime_error("[CLDNN] clEnqueueMemcpyINTEL is nullptr");
//         cl_event tmp;
//         cl_int err = _enqueue_memcpy_fn(
//             cpp_queue.get(),
//             static_cast<cl_bool>(blocking),
//             dst_ptr,
//             src_ptr,
//             bytes_count,
//             wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
//             wait_list == nullptr ? nullptr : reinterpret_cast<const cl_event*>(&wait_list->front()),
//             ret_event == nullptr ? nullptr : &tmp);

//         if (ret_event != nullptr && err == CL_SUCCESS)
//             *ret_event = tmp;

//         return err;
//     }

//     cl_int enqueue_fill_mem(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void* pattern,
//                             size_t pattern_size, size_t bytes_count, const std::vector<cl::Event>* wait_list = nullptr,
//                             cl::Event* ret_event = nullptr) const {
//         if (!_enqueue_mem_fill_fn)
//             throw std::runtime_error("[CLDNN] clEnqueueMemFillINTEL is nullptr");
//         cl_event tmp;
//         cl_int err = _enqueue_mem_fill_fn(
//             cpp_queue.get(),
//             dst_ptr,
//             pattern,
//             pattern_size,
//             bytes_count,
//             wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
//             wait_list == nullptr ? nullptr :  reinterpret_cast<const cl_event*>(&wait_list->front()),
//             ret_event == nullptr ? nullptr : &tmp);

//         if (ret_event != nullptr && err == CL_SUCCESS)
//             *ret_event = tmp;

//         return err;
//     }

//     cl_int enqueue_set_mem(const cl::CommandQueue& cpp_queue, void* dst_ptr, cl_int value,
//                            size_t bytes_count, const std::vector<cl::Event>* wait_list = nullptr,
//                            cl::Event* ret_event = nullptr) const {
//         if (!_enqueue_memset_fn)
//             throw std::runtime_error("[CLDNN] clEnqueueMemsetINTEL is nullptr");
//         cl_event tmp;
//         cl_int err = _enqueue_memset_fn(
//             cpp_queue.get(),
//             dst_ptr,
//             value,
//             bytes_count,
//             wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
//             wait_list == nullptr ? nullptr :  reinterpret_cast<const cl_event*>(&wait_list->front()),
//             ret_event == nullptr ? nullptr : &tmp);

//         if (ret_event != nullptr && err == CL_SUCCESS)
//             *ret_event = tmp;

//         return err;
//     }

//     cl_unified_shared_memory_type_intel get_usm_allocation_type(const void* usm_ptr) const {
//         if (!_get_mem_alloc_info_fn) {
//             throw std::runtime_error("[GPU] clGetMemAllocInfoINTEL is nullptr");
//         }

//         cl_unified_shared_memory_type_intel ret_val;
//         size_t ret_val_size;
//         _get_mem_alloc_info_fn(_ctx.get(), usm_ptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(cl_unified_shared_memory_type_intel), &ret_val, &ret_val_size);
//         return ret_val;
//     }

//     size_t get_usm_allocation_size(const void* usm_ptr) const {
//         if (!_get_mem_alloc_info_fn) {
//             throw std::runtime_error("[GPU] clGetMemAllocInfoINTEL is nullptr");
//         }

//         size_t ret_val;
//         size_t ret_val_size;
//         _get_mem_alloc_info_fn(_ctx.get(), usm_ptr, CL_MEM_ALLOC_SIZE_INTEL, sizeof(size_t), &ret_val, &ret_val_size);
//         return ret_val;
//     }

// private:
//     cl::Context _ctx;
//     cl::Device _device;
//     clHostMemAllocINTEL_fn _host_mem_alloc_fn = nullptr;
//     clMemFreeINTEL_fn _mem_free_fn = nullptr;
//     clSharedMemAllocINTEL_fn _shared_mem_alloc_fn = nullptr;
//     clDeviceMemAllocINTEL_fn _device_mem_alloc_fn = nullptr;
//     clSetKernelArgMemPointerINTEL_fn _set_kernel_arg_mem_pointer_fn = nullptr;
//     clEnqueueMemcpyINTEL_fn _enqueue_memcpy_fn = nullptr;
//     clEnqueueMemFillINTEL_fn _enqueue_mem_fill_fn = nullptr;
//     clEnqueueMemsetINTEL_fn _enqueue_memset_fn = nullptr;
//     clGetMemAllocInfoINTEL_fn _get_mem_alloc_info_fn = nullptr;
// };

#else
using BufferType = void*;
#endif

std::map<std::string, ov::TensorVector> get_remote_input_tensors(
    const std::map<std::string, std::vector<std::string>>& inputFiles,
    const std::vector<benchmark_app::InputsInfo>& app_inputs_info,
    const ov::CompiledModel& compiledModel,
    std::vector<BufferType>& clBuffer,
    size_t num_requests);

std::map<std::string, ov::Tensor> get_remote_output_tensors(const ov::CompiledModel& compiledModel,
                                                            std::map<std::string, ::gpu::BufferType>& clBuffer);
}  // namespace gpu
