// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for internal properties that are passed from one plugin to another
 * @file openvino/runtime/internal_properties.hpp
 */

#pragma once

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {

namespace internal {
/**
 * @brief Read-only property to get a std::vector<PropertyName> of supported internal properties.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> supported_properties{
    "INTERNAL_SUPPORTED_PROPERTIES"};

/**
 * @brief Read-only property to get a std::vector<PropertyName> of properties
 * which should affect the hash calculation for model cache
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> caching_properties{"CACHING_PROPERTIES"};

/**
 * @brief Allow to create exclusive_async_requests with one executor
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RW> exclusive_async_requests{"EXCLUSIVE_ASYNC_REQUESTS"};

/**
 * @brief the property for setting of required device for which config to be updated
 * values: device id starts from "0" - first device, "1" - second device, etc
 * note: plugin may have different devices naming convention
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::string, PropertyMutability::WO> config_device_id{"CONFIG_DEVICE_ID"};

/**
 * @brief Limit \#threads that are used by IStreamsExecutor to execute `parallel_for` calls
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<int32_t, PropertyMutability::RW> threads_per_stream{"THREADS_PER_STREAM"};

/**
 * @brief It contains compiled_model_runtime_properties information to make plugin runtime can check whether it is
 * compatible with the cached compiled model, the result is returned by get_property() calling.
 *
 * The information details are defined by plugin itself, each plugin may require different runtime contents.
 * For example, CPU plugin will contain OV version, while GPU plugin will contain OV and GPU driver version, etc.
 * Core doesn't understand its content and only read it from plugin and write it into blob header.
 *
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::string, PropertyMutability::RO> compiled_model_runtime_properties{
    "COMPILED_MODEL_RUNTIME_PROPERTIES"};

/**
 * @brief Check whether the attached compiled_model_runtime_properties is supported by this device runtime.
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RO> compiled_model_runtime_properties_supported{
    "COMPILED_MODEL_RUNTIME_PROPERTIES_SUPPORTED"};

/**
 * @brief Read-write property to set the percentage of the estimated model size which is used to determine the query
 * model results for further processing
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<float, PropertyMutability::RW> query_model_ratio{"QUERY_MODEL_RATIO"};

/**
 * @brief Namespace for properties related to MLIR operations within the GPU plugin.
 * These properties are used as evaluation context parameters for MLIR operations,
 * assisting in managing events, result tracking, and kernel argument types.
 */
namespace mlir_meta {

/**
 * @brief This key identifies a list of cl_event to wait for a kernel execution.
 * @ingroup ov_dev_api_plugin_mlir_meta_api
 */
static constexpr Property<std::vector<void*>> wait_list{"EVENTS_WAIT_LIST"};

/**
 * @brief This key identifies a pointer to a cl::Event that should be set with
 * the result cl_event of a kernel execution. Example:
 * @code
 *     cl_event result_event = launchModuleAndGetEvent();
 *     cl::Event* ev = evaluationContext[ov::internal::mlir_meta::result_event.name()].as<cl::Event*>();
 *     *ev = result_event; // cl::Event::operator=() will retain the event
 * @ingroup ov_dev_api_plugin_mlir_meta_api
 */
static constexpr Property<void*> result_event{"RESULT_EVENT"};

/**
 * @brief This key identifies whether the kernel argument at [i] position is USM pointer
 * @ingroup ov_dev_api_plugin_mlir_meta_api
 */
static constexpr Property<std::vector<bool>> is_kernel_arg_usm{"IS_KERNEL_ARG_USM"};

} // namespace mlir_meta
}  // namespace internal
}  // namespace ov
