/**
 * @file multi_wrapper_impls.hpp
 *
 * @brief Implementations of methods or functions requiring the definitions of
 * multiple CUDA entity proxy classes. In some cases these are declared in the
 * individual proxy class files, with the other classes forward-declared.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_HPP_
#define MULTI_WRAPPER_IMPLS_HPP_

#include <cuda/api/array.hpp>
#include <cuda/api/device.hpp>
#include <cuda/api/event.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/kernel.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/stream.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda/api/texture_view.hpp>
#include <cuda_runtime.h>

#include <type_traits>
#include <vector>
#include <algorithm>

namespace cuda {

template <typename T, dimensionality_t NumDimensions>
device_t array_t<T, NumDimensions>::device() const noexcept
{
    return device::get(device_id_);
}

template <typename T, dimensionality_t NumDimensions>
texture_view::texture_view(
    const cuda::array_t<T, NumDimensions>&  arr,
    texture::descriptor_t                   descriptor)
    : device_id_(arr.device().id()), owning(true)
{
    cudaResourceDesc resource_descriptor;
    memset(&resource_descriptor, 0, sizeof(resource_descriptor));
    resource_descriptor.resType = cudaResourceTypeArray;
    resource_descriptor.res.array.array = arr.get();

    auto status = cudaCreateTextureObject(&raw_handle_, &resource_descriptor, &descriptor, nullptr);
    throw_if_error(status, "failed creating a CUDA texture object");
}

inline device_t texture_view::associated_device() const noexcept
{
    return cuda::device::get(device_id_);
}

namespace array {

namespace detail_ {

template <typename T, dimensionality_t NumDimensions>
handle_t create(const device_t& device, dimensions_t<NumDimensions> dimensions)
{
	return create<T, NumDimensions>(device.id(), dimensions);
}

} // namespace detail_

template <typename T, dimensionality_t NumDimensions>
array_t<T, NumDimensions> create(
    const device_t&              device,
    dimensions_t<NumDimensions>  dimensions)
{
    handle_t handle { detail_::create<T, NumDimensions>(device, dimensions) };
    return wrap<T>(device.id(), handle, dimensions);
}

} // namespace array

namespace event {

inline event_t create(
	device_t   device,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	auto device_id = device.id();
		// Yes, we need the ID explicitly even on the current device,
		// because event_t's don't have an implicit device ID.
	return event::detail_::create(device_id , uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(event_t& event)
{
	return detail_::export_(event.handle());
}

inline event_t import(device_t& device, const handle_t& handle)
{
	return event::detail_::wrap(device.id(), detail_::import(handle), do_not_take_ownership);
}

} // namespace ipc

} // namespace event


// device_t methods

inline stream_t device_t::default_stream() const noexcept
{
	return stream::detail_::wrap(id(), stream::default_stream_handle);
}

inline stream_t
device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority) const
{
	device::current::detail_::scoped_override_t set_device_for_this_scope(id_);
	return stream::detail_::wrap(id(), stream::detail_::create_on_current_device(
		will_synchronize_with_default_stream, priority), do_take_ownership);
}

namespace device {
namespace current {

inline scoped_override_t::scoped_override_t(device_t& device) : parent(device.id()) { }
inline scoped_override_t::scoped_override_t(device_t&& device) : parent(device.id()) { }

} // namespace current
} // namespace device


namespace detail_ {

} // namespace detail_

template <typename KernelFunction, typename ... KernelParameters>
void device_t::launch(
	KernelFunction kernel_function, launch_configuration_t launch_configuration,
	KernelParameters ... parameters) const
{
	return default_stream().enqueue.kernel_launch(
		kernel_function, launch_configuration, parameters...);
}

inline event_t device_t::create_event(
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess) const
{
	// The current implementation of event::create is not super-smart,
	// but it's probably not worth it trying to improve just this function
	return event::create(*this, uses_blocking_sync, records_timing, interprocess);
}

// event_t methods

inline device_t event_t::device() const noexcept
{
	return cuda::device::get(device_id_);
}

inline void event_t::record(const stream_t& stream)
{
	// Note:
	// TODO: Perhaps check the device ID here, rather than
	// have the Runtime API call fail?
	event::detail_::enqueue(stream.handle(), handle_);
}

inline void event_t::fire(const stream_t& stream)
{
	record(stream);
	stream.synchronize();
}

// stream_t methods

inline device_t stream_t::device() const noexcept
{
	return cuda::device::get(device_id_);
}

inline void stream_t::enqueue_t::wait(const event_t& event)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id);

	// Required by the CUDA runtime API; the flags value is currently unused
	constexpr const unsigned int flags = 0;

	auto status = cudaStreamWaitEvent(associated_stream.handle_, event.handle(), flags);
	throw_if_error(status,
		::std::string("Failed scheduling a wait for " + event::detail_::identify(event.handle())
		+ " on stream " + stream::detail_::identify(associated_stream.handle_, associated_stream.device_id_)));

}

inline event_t& stream_t::enqueue_t::event(event_t& existing_event)
{
	auto device_id = associated_stream.device_id_;
	if (existing_event.device_id() != device_id) {
		throw ::std::invalid_argument("Attempt to enqueue a CUDA event associated with "
			+ device::detail_::identify(existing_event.device_id()) + " to be triggered by a stream on "
			+ device::detail_::identify(device_id));
	}
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id);
	stream::detail_::record_event_on_current_device(device_id, associated_stream.handle_, existing_event.handle());
	return existing_event;
}

inline event_t stream_t::enqueue_t::event(
    bool          uses_blocking_sync,
    bool          records_timing,
    bool          interprocess)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);

	event_t ev { event::detail_::create_on_current_device(device_id, uses_blocking_sync, records_timing, interprocess) };
	// Note that, at this point, the event is not associated with this enqueue object's stream.
	stream::detail_::record_event_on_current_device(device_id, associated_stream.handle_, ev.handle());
	return ev;
}

namespace memory {

template <typename T>
inline device_t pointer_t<T>::device() const noexcept
{
	return cuda::device::get(attributes().device);
}

namespace async {

inline void copy(void *destination, const void *source, size_t num_bytes, const stream_t& stream)
{
	detail_::copy(destination, source, num_bytes, stream.handle());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream)
{
	detail_::copy(destination, source, stream.handle());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(T* destination, const array_t<T, NumDimensions>& source, const stream_t& stream)
{
	detail_::copy(destination, source, stream.handle());
}

template <typename T>
inline void copy_single(T& destination, const T& source, const stream_t& stream)
{
	detail_::copy_single(&destination, &source, sizeof(T), stream.handle());
}

} // namespace async

namespace device {

inline region_t allocate(cuda::device_t device, size_t size_in_bytes)
{
	return detail_::allocate(device.id(), size_in_bytes);
}

namespace async {

inline region_t allocate(const stream_t& stream, size_t size_in_bytes)
{
	return detail_::allocate(stream.device().id(), stream.handle(), size_in_bytes);
}

inline void set(void* start, int byte_value, size_t num_bytes, const stream_t& stream)
{
	detail_::set(start, byte_value, num_bytes, stream.handle());
}

inline void zero(void* start, size_t num_bytes, const stream_t& stream)
{
	detail_::zero(start, num_bytes, stream.handle());
}

} // namespace async

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * the current device's global memory
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param num_elements  the number of elements to allocate
 * @return an ::std::unique_ptr pointing to the constructed T array
 */
template<typename T>
inline unique_ptr<T> make_unique(size_t num_elements)
{
	static_assert(::std::is_array<T>::value, "make_unique<T>(device, num_elements) can only be invoked for T being an array type, T = U[]");
	return cuda::memory::detail_::make_unique<T, detail_::allocator, detail_::deleter>(num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param device        on which to construct the array of elements
 * @param num_elements  the number of elements to allocate
 * @return an ::std::unique_ptr pointing to the constructed T array
 */template<typename T>
inline unique_ptr<T> make_unique(device_t device, size_t num_elements)
{
    cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device.id());
    return make_unique<T>(num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in the current device's global memory
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @return an ::std::unique_ptr pointing to the allocated memory
 */
template <typename T>
inline unique_ptr<T> make_unique()
{
	return cuda::memory::detail_::make_unique<T, detail_::allocator, detail_::deleter>();
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in device-global memory
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @param device  on which to construct the T element
 * @return an ::std::unique_ptr pointing to the allocated memory
 */
template <typename T>
inline unique_ptr<T> make_unique(device_t device)
{
    cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device.id());
    return make_unique<T>();
}

} // namespace device

namespace managed {

namespace detail_ {

template <typename T>
inline device_t base_region_t<T>::preferred_location() const
{
	auto device_id = detail_::get_scalar_range_attribute<bool>(*this, cudaMemRangeAttributePreferredLocation);
	return cuda::device::get(device_id);
}

template <typename T>
inline void base_region_t<T>::set_preferred_location(device_t& device) const
{
	detail_::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseSetPreferredLocation, device.id());
}

template <typename T>
inline void base_region_t<T>::clear_preferred_location() const
{
	detail_::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseUnsetPreferredLocation);
}

} // namespace detail_


inline void advise_expected_access_by(const_region_t region, device_t& device)
{
	detail_::set_scalar_range_attribute(region, cudaMemAdviseSetAccessedBy, device.id());
}

inline void advise_no_access_expected_by(const_region_t region, device_t& device)
{
	detail_::set_scalar_range_attribute(region, cudaMemAdviseUnsetAccessedBy, device.id());
}

template <typename Allocator>
::std::vector<device_t, Allocator> accessors(const_region_t region, const Allocator& allocator)
{
	static_assert(sizeof(cuda::device::id_t) == sizeof(device_t), "Unexpected size difference between device IDs and their wrapper class, device_t");

	auto num_devices = cuda::device::count();
	::std::vector<device_t, Allocator> devices(num_devices, allocator);
	auto device_ids = reinterpret_cast<cuda::device::id_t *>(devices.data());


	auto status = cudaMemRangeGetAttribute(
		device_ids, sizeof(device_t) * devices.size(),
		cudaMemRangeAttributeAccessedBy, region.start(), region.size());
	throw_if_error(status, "Obtaining the IDs of devices with access to the managed memory range at " + cuda::detail_::ptr_as_hex(region.start()));
	auto first_invalid_element = ::std::lower_bound(device_ids, device_ids + num_devices, cudaInvalidDeviceId);
	// We may have gotten less results that the set of all devices, so let's whittle that down

	if (first_invalid_element - device_ids != num_devices) {
		devices.resize(first_invalid_element - device_ids);
	}

	return devices;
}

namespace async {

inline void prefetch(
	const_region_t   region,
	cuda::device_t   destination,
	const stream_t&  stream)
{
	detail_::prefetch(region, destination.id(), stream.handle());
}

} // namespace async


inline region_t allocate(
	cuda::device_t        device,
	size_t                size_in_bytes,
	initial_visibility_t  initial_visibility)
{
	return detail_::allocate(device.id(), size_in_bytes, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
    device_t              device,
    size_t                num_elements,
    initial_visibility_t  initial_visibility)
{
    cuda::device::current::detail_::scoped_override_t(device.id());
    return make_unique<T>(num_elements, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
    device_t              device,
    initial_visibility_t  initial_visibility)
{
    cuda::device::current::detail_::scoped_override_t(device.id());
    return make_unique<T>(initial_visibility);
}

} // namespace managed

namespace mapped {

inline region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options)
{
	return cuda::memory::mapped::detail_::allocate(device.id(), size_in_bytes, options);
}

} // namespace mapped

} // namespace memory

// kernel_t methods

inline device_t kernel_t::device() const noexcept { return device::get(device_id_); }

inline void kernel_t::set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	auto result = cudaFuncSetAttribute(ptr_, attribute, value);
	throw_if_error(result, "Setting CUDA device function attribute " + ::std::to_string(attribute) + " to value " + ::std::to_string(value));
}

inline void kernel_t::opt_in_to_extra_dynamic_memory(cuda::memory::shared::size_t amount_required_by_kernel)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
#if CUDART_VERSION >= 9000
	auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributeMaxDynamicSharedMemorySize, amount_required_by_kernel);
	throw_if_error(result,
		"Trying to opt-in to " + ::std::to_string(amount_required_by_kernel) + " bytes of dynamic shared memory, "
		"exceeding the maximum available on device " + ::std::to_string(device_id_) + " (consider the amount of static shared memory"
		"in use by the function).");
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

#if defined(__CUDACC__)

// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

namespace detail_ {

template <typename UnaryFunction>
inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const void *             ptr,
	device::id_t             device_id,
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
#if CUDART_VERSION <= 10000
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#else
	int min_grid_size_in_blocks { 0 };
	int block_size { 0 };
		// Note: only initializing the values her because of a
		// spurious (?) compiler warning about potential uninitialized use.
	auto result = cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr,
		block_size_to_dynamic_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
	);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail_::ptr_as_hex(ptr) +
			" on device " + ::std::to_string(device_id) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif // CUDART_VERSION <= 10000
}

inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const void *             ptr,
	device::id_t             device_id,
	memory::shared::size_t   dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	auto always_need_same_shared_mem_size =
		[dynamic_shared_mem_size](::size_t) { return dynamic_shared_mem_size; };
	return min_grid_params_for_max_occupancy(
		ptr, device_id, always_need_same_shared_mem_size, block_size_limit, disable_caching_override);
}

} // namespace detail_


inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const kernel_t&          kernel,
	memory::shared::size_t   dynamic_shared_memory_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	return detail_::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device().id(), dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}


inline grid::complete_dimensions_t kernel_t::min_grid_params_for_max_occupancy(
	memory::shared::size_t   dynamic_shared_memory_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override) const
{
	return detail_::min_grid_params_for_max_occupancy(
		ptr_, device_id_, dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}

template <typename UnaryFunction>
grid::complete_dimensions_t kernel_t::min_grid_params_for_max_occupancy(
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override) const
{
	return detail_::min_grid_params_for_max_occupancy(
		ptr_, device_id_, block_size_to_dynamic_shared_mem_size, block_size_limit, disable_caching_override);
}

#endif // defined __CUDACC__

inline void kernel_t::set_preferred_shared_mem_fraction(unsigned shared_mem_percentage)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	if (shared_mem_percentage > 100) {
		throw ::std::invalid_argument("Percentage value can't exceed 100");
	}
#if CUDART_VERSION >= 9000
	auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributePreferredSharedMemoryCarveout, shared_mem_percentage);
	throw_if_error(result, "Trying to set the carve-out of shared memory/L1 cache memory");
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif // CUDART_VERSION <= 9000
}

inline kernel::attributes_t kernel_t::attributes() const
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	kernel::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void kernel_t::set_cache_preference(multiprocessor_cache_preference_t  preference)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}


inline void kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
	throw_if_error(result);
}

inline grid::dimension_t kernel_t::maximum_active_blocks_per_multiprocessor(
	grid::block_dimension_t   num_threads_per_block,
	memory::shared::size_t    dynamic_shared_memory_per_block,
	bool                      disable_caching_override)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	int result;
	unsigned int flags = disable_caching_override ?
		cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, ptr_, (int) num_threads_per_block,
		dynamic_shared_memory_per_block, flags);
	throw_if_error(status, "Failed calculating the maximum occupancy "
		"of device function blocks per multiprocessor");
	return result;
}


namespace kernel {

template<typename KernelFunctionPtr>
kernel_t wrap(const device_t &device, KernelFunctionPtr function_ptr)
{
	static_assert(
		::std::is_pointer<KernelFunctionPtr>::value
			and ::std::is_function<typename ::std::remove_pointer<KernelFunctionPtr>::type>::value,
		"function_ptr must be a bona fide pointer to a kernel (__global__) function");
	return detail_::wrap(device.id(), reinterpret_cast<const void*>(function_ptr));
}

} // namespace kernel

namespace stream {

inline stream_t create(
	device_t    device,
	bool        synchronizes_with_default_stream,
	priority_t  priority)
{
	return detail_::create(device.id(), synchronizes_with_default_stream, priority);
}

namespace detail_ {

inline void record_event_on_current_device(device::id_t device_id, stream::handle_t stream_handle, event::handle_t event_handle)
{
	auto status = cudaEventRecord(event_handle, stream_handle);
	throw_if_error(status,
		"Failed scheduling " + event::detail_::identify(event_handle) + " to occur"
		+ " on stream " + stream::detail_::identify(stream_handle, device_id));
}
} // namespace detail_

} // namespace stream

namespace detail_ {

template<typename... KernelParameters>
inline void enqueue_launch(
	::std::true_type,       // got a wrapped kernel
	const kernel_t&         kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	if (kernel.device() != stream.device()) {
		throw ::std::invalid_argument("Attempt to enqueue a kernel for "
			+ device::detail_::identify(kernel.device().id()) + " on a stream for "
			+ "device " + device::detail_::identify(stream.device().id()));
	}

	// Note: We are not performing an imperfect un-erasure of the wrapper function pointer's
	// signature. Imperfect - since the KernelParameter pack may contain some
	// references, arrays and so on - which CUDA kernels cannot accept; so have
	// to massage it a bit.

	using raw_kernel_type = void(*)(typename cuda::detail_::kernel_parameter_decay_t<KernelParameters>...);
//		typename cuda::kernel::detail_::raw_kernel_typegen<
//			typename cuda::detail_::kernel_parameter_decay_t<KernelParameters>::type ...
//		>::type;

	auto unwrapped_kernel_function = reinterpret_cast<raw_kernel_type>(const_cast<void*>(kernel.ptr()));

	detail_::enqueue_launch(
		unwrapped_kernel_function,
		stream.handle(),
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template<typename RawKernelFunction, typename... KernelParameters>
inline void enqueue_launch(
	::std::false_type,      // got a raw function
	RawKernelFunction       kernel_function,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	static_assert(
		::std::is_function<typename ::std::decay<RawKernelFunction>::type>::value
		or (
			::std::is_pointer<RawKernelFunction>::value
			and ::std::is_function<typename ::std::remove_pointer<RawKernelFunction>::type>::value
			)
		, "Invalid Kernel type - it must be either a function or a pointer-to-a-function");


	// Note: It is possible that the parameter pack's signature does not exactly fit the function
	// pointer - it contain some references, arrays and so on - which CUDA kernels cannot accept;
	// but it should be close enough to pass to the function... or the user would not have asked
	// to launch the kernel this way. So, no reinterpretation/decay should be necessary here.

	detail_::enqueue_launch(
		kernel_function,
		stream.handle(),
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}


} // namespace detail_

template<typename Kernel, typename... KernelParameters>
inline void enqueue_launch(
	Kernel                  kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	constexpr const auto got_a_wrapped_kernel = ::std::is_same<typename ::std::decay<Kernel>::type, kernel_t>::value;
	using got_a_wrapped_kernel_type = ::std::integral_constant<bool, got_a_wrapped_kernel>;
	return detail_::enqueue_launch(got_a_wrapped_kernel_type{}, kernel, stream, launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

namespace detail_ {

template<typename Kernel>
device_t get_implicit_device(Kernel)
{
	return device::current::get();
}

template<>
inline device_t get_implicit_device<kernel_t>(kernel_t kernel)
{
	return kernel.device();
}

} // namespace detail_
template<typename Kernel, typename... KernelParameters>
inline void launch(
	Kernel                  kernel,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	auto device = detail_::get_implicit_device(kernel);
	stream_t stream = device.default_stream();

	// Note: If Kernel is a kernel_t, and its associated device is different
	// than the current device, the next call will fail:

	enqueue_launch(
		kernel,
		stream,
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}


} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_HPP_

