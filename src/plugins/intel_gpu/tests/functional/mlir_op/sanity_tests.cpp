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
#include "openvino/core/preprocess/pre_post_process.hpp"

using testing::ElementsAreArray;

static std::string model_full_path(const char* path) {
    return ov::util::make_path<char>(TEST_MODELS_DIR, path);
}

void multiply_matrices_and_add_a(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b,
                       std::vector<float>& result, int rows_a, int cols_a, int cols_b) {
    // Initialize the result matrix with zero values
    std::fill(result.begin(), result.end(), 0.0f);

    // Matrix multiplication logic using linear indexing
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            for (int k = 0; k < cols_a; ++k) {
                result[i * cols_b + j] += matrix_a[i * cols_a + k] * matrix_b[k * cols_b + j];
            }
        }
    }
    for (int i = 0; i < matrix_a.size(); i++) {
        result[i] += matrix_a[i];
    }
}

std::vector<float> read_float_array_from_binary_file(const std::string& filename) {
    // Open the binary file in input mode and binary mode
    std::ifstream input_file(filename, std::ios::binary);

    // Check if the file was successfully opened
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    // Move the cursor to the end to determine the size of the file
    input_file.seekg(0, std::ios::end);
    std::streamsize file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    // Calculate the number of floats in the file
    std::size_t num_floats = file_size / sizeof(float);

    // Create a vector to store the floats
    std::vector<float> float_array(num_floats);

    // Read the floats from the file into the vector
    if (num_floats > 0) {
        input_file.read(reinterpret_cast<char*>(float_array.data()), file_size);
    }

    // Close the file
    input_file.close();

    return float_array;
}

template<typename T>
static ov::Tensor allocate_usm_tensor(
        ov::intel_gpu::ocl::ClContext& oclContext, OpenCL* oclInstance, const ov::Shape& shape,
        ov::element::Type type, std::vector<T> &input_values) {
    cl_int err;
    size_t byte_size = shape_size(shape) * type.bitwidth() / 8;

    void* usm_ptr = oclInstance->_usm_helper->allocate_device(
        /*properties=*/nullptr,
        /*size=*/byte_size,
        /*alignment=*/0,
        /*err_code_return=*/&err);
    std::cout << "allocated: " << usm_ptr << std::endl;

    cl_event tmp;
    err = oclInstance->_usm_helper->enqueue_memcpy(
        oclInstance->_queue,
        /*dst=*/usm_ptr,
        /*src=*/input_values.data(),
        byte_size,
        /*blocking=*/true,
        /*wait_list=*/nullptr,
        /*ret_event=*/nullptr);

    return oclContext.create_tensor(type, shape, usm_ptr);
}

template<typename T>
std::vector<T> broadcast_vector(const std::vector<T>& v, size_t new_size) {
    std::vector<T> result;
    result.reserve(new_size);

    size_t original_size = v.size();

    if (original_size == 0) {
        throw std::invalid_argument("Original vector size must be greater than 0.");
    }

    // Fill the result vector by repeating the input vector
    for (size_t i = 0; i < new_size; ++i) {
        result.push_back(v[i % original_size]);
    }

    return result;
}

template<typename T>
static std::map<size_t, ov::Tensor> allocate_input_tensors(
        ov::CompiledModel& compiledModel,
        std::map<size_t, std::vector<T>> &inputValues) {
    auto context = compiledModel.get_context();
    auto& oclContext = static_cast<ov::intel_gpu::ocl::ClContext&>(context);
    auto oclInstance = std::make_shared<OpenCL>(oclContext.get());

    std::map<size_t, ov::Tensor> input_tensors;
    for (const auto& input : compiledModel.inputs()) {
        auto shape = input.get_shape();
        auto size = ov::shape_size(shape);
        std::vector<T> input_values = broadcast_vector(inputValues[input.get_index()], size);
        auto tensor = allocate_usm_tensor(oclContext, oclInstance.get(), shape, input.get_element_type(), input_values);
        input_tensors.emplace(input.get_index(), tensor);
    }
    return input_tensors;
}

TEST(MLIRExecution, SimpleMatmulf32) {
    ov::Core core;
    auto model = core.read_model(
        model_full_path("matmul_64_128_f32.xml"));

    ov::AnyMap device_config;
    device_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    device_config[ov::enable_profiling.name()] = false;
    device_config.emplace(ov::hint::inference_precision("f32"));

    auto compiled_model = core.compile_model(model, "GPU", device_config);

    std::map<size_t, std::vector<float>> input_values_map;
    input_values_map.emplace(0, std::vector<float>(1, 0.5f));

    auto input_tensors = allocate_input_tensors(compiled_model, input_values_map);

    auto infer_req = compiled_model.create_infer_request();
    for (const auto& input : input_tensors) {
        infer_req.set_input_tensor(input.first, input.second);
    }
    infer_req.infer();

    auto computed = infer_req.get_output_tensor(0);
    float* result = reinterpret_cast<float*>(computed.data());

    // compute reference result
    std::vector<float> matrix_a = broadcast_vector(input_values_map.at(0), 64 * 128);
    std::vector<float> matrix_b = read_float_array_from_binary_file(model_full_path("matmul_64_128_f32.bin"));
    ASSERT_EQ(matrix_b.size(), 128 * 128);
    std::vector<float> reference_result(64 * 128);
    multiply_matrices_and_add_a(matrix_a, matrix_b, reference_result, 64, 128, 128);

    // compare result with the reference
    for (size_t i = 0; i < reference_result.size(); ++i) {
        EXPECT_NEAR(reference_result[i], result[i], 1e-5);
    }
}
