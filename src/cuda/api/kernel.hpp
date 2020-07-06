/**
 * @file kernel.hpp
 *
 * @brief Contains a base wrapper class for CUDA kernels - both statically and
 * dynamically compiled; and some related functionality.
 *
 * @note This file does _not_ define any kernels itself.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_HPP_
#define CUDA_API_WRAPPERS_KERNEL_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/current_context.hpp>
// #include <cuda/api/module.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace cuda {

///@cond
class device_t;
class kernel_t;
///@nocond

namespace kernel {

namespace detail_ {

kernel_t wrap(device::id_t device_id, context::handle_t context_id,	kernel::handle_t f);

#ifndef NDEBUG
static const char* attribute_name(int attribute_index)
{
	// Note: These correspond to the values of enum CUfunction_attribute_enum
	static const char* names[] = {
		"Maximum number of threads per block",
		"Statically-allocated shared memory size in bytes",
		"Required constant memory size in bytes",
		"Required local memory size in bytes",
		"Number of registers used by each thread",
		"PTX virtual architecture version into which the kernel code was compiled",
		"Binary architecture version for which the function was compiled",
		"Indication whether the function was compiled with cache mode CA",
		"Maximum allowed size of dynamically-allocated shared memory use size bytes",
		"Preferred shared memory carve-out to actual shared memory"
	};
	return names[attribute_index];
}
#endif

inline attribute_value_t get_attribute_in_current_context(handle_t handle, attribute_t attribute)
{
	kernel::attribute_value_t attribute_value;
	auto result = cuFuncGetAttribute(&attribute_value,  attribute, handle);
	throw_if_error(result,
		::std::string("Failed obtaining attribute ") +
#ifdef NDEBUG
			::std::to_string(static_cast<::std::underlying_type<kernel::attribute_t>::type>(attribute))
#else
			attribute_name(attribute)
#endif
	);
	return attribute_value;
}

} // namespace detail_

} // namespace kernel

/**
 * A non-owning wrapper for CUDA kernels - whether they be `__global__` functions compiled
 * apriori, or the result of dynamic NVRTC compilation, or obtained in some other future
 * way.
 *
 * @note The association of a `kernel_t` with an individual device or context is somewhat
 * tenuous. That is, the same function could be used with any other compatible device;
 * However, many/most of the features, attributes and settings are context-specific
 * or device-specific.
 *
 * @note NVRTC-compiled kernels can only use this class, with apriori-compiled
 * kernels can use their own subclass.
 */
class kernel_t {

public: // getters
	context_t context() const noexcept;
	device_t device() const noexcept;

	device::id_t      device_id()  const noexcept { return device_id_; }
	context::handle_t context_handle() const noexcept { return context_handle_; }
	kernel::handle_t  handle()     const noexcept { return handle_; }

public: // non-mutators

	kernel::attribute_value_t get_attribute(kernel::attribute_t attribute) const
	{
		context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
		return kernel::detail_::get_attribute_in_current_context(handle(), attribute);
	}

	cuda::device::compute_capability_t ptx_version() const noexcept {
		auto raw_attribute = get_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
		return device::compute_capability_t::from_combined_number(raw_attribute);
	}

	cuda::device::compute_capability_t binary_compilation_target_architecture() const noexcept {
		auto raw_attribute = get_attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
		return device::compute_capability_t::from_combined_number(raw_attribute);
	}

	/**
	 * @return the maximum number of threads per block for which the GPU device can satisfy
	 * this kernel's hardware requirement - typically, the number of registers in use.
	 *
	 * @note the kernel may have other constraints, requiring a different number of threads
	 * per block; these cannot be determined using this method.
	 */
	grid::block_dimension_t maximum_threads_per_block() const
	{
		return get_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	}

public: // methods mutating the kernel-in-context, but not this reference object

	void set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const;

	/**
	 * @brief Change the hardware resource carve-out between L1 cache and shared memory
	 * for launches of the kernel to allow for at least the specified amount of
	 * shared memory.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but can
	 * also be set on the individual device-function level, by specifying the amount of shared
	 * memory the kernel may require.
	 */
	void set_maximum_dynamic_shared_memory_per_block(cuda::memory::shared::size_t amount_required_by_kernel) const
	{
		auto amount_required_by_kernel_ = (kernel::attribute_value_t) amount_required_by_kernel;
		if (amount_required_by_kernel != (cuda::memory::shared::size_t) amount_required_by_kernel_) {
			throw ::std::invalid_argument("Requested amount of maximum shared memory exceeds the "
				"representation range for kernel attribute values");
		}
		// TODO: Consider a check in debug mode for the value being within range
		set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,amount_required_by_kernel_);
	}

	/**
	 * @brief Indicate the desired carve-out between shared memory and L1 cache when launching
	 * this kernel - with coarse granularity.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but the
	 * driver can set another value for a specific function. This function doesn't make a demand
	 * from the CUDA runtime (as in @p opt_in_to_extra_dynamic_memory), but rather indicates
	 * what is the fraction of L1 to shared memory it would like the kernel scheduler to carve
	 * out.
	 *
	 * @param preference one of: as much shared memory as possible, as much
	 * L1 as possible, or no preference (i.e. using the device default).
	 *
	 * @note similar to @ref set_preferred_shared_mem_fraction() - but with coarser granularity.
	 */
	void set_cache_preference(multiprocessor_cache_preference_t preference)
	{
		context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
		auto result = cuFuncSetCacheConfig(handle(), (CUfunc_cache) preference);
		throw_if_error(result,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
			"CUDA device function");
	}

	/**
	 * @brief Sets a device function's preference of shared memory bank size
	 *
	 * @param config bank size setting to make
	 */
	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config)
	{
		// TODO: Need to set a context, not a device
		context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
		auto result = cuFuncSetSharedMemConfig(handle(), static_cast<CUsharedconfig>(config) );
		throw_if_error(result, "Failed setting the shared memory bank size");
	}

protected: // ctors & dtor
	kernel_t(device::id_t device_id, context::handle_t context_handle, kernel::handle_t handle = nullptr)
		: device_id_(device_id), context_handle_(context_handle), handle_(handle) { }

public: // ctors & dtor
	friend kernel_t kernel::detail_::wrap(device::id_t, context::handle_t, kernel::handle_t);

	kernel_t(const kernel_t& other) = default; // Note: be careful with subclasses
	kernel_t(kernel_t&& other) = default; // Note: be careful with subclasses

public: // ctors & dtor
	virtual ~kernel_t() = default;

protected: // data members
	device::id_t device_id_; // We don't _absolutely_ need the device ID, but - why not have it if we can?
	context::handle_t context_handle_;
	mutable kernel::handle_t handle_;
}; // kernel_t

namespace kernel {

namespace detail_ {

inline kernel_t wrap(
	device::id_t       device_id,
	context::handle_t  context_id,
	kernel::handle_t   f)
{
	return kernel_t{ device_id, context_id, f };
}

} // namespace detail_

} // namespace kernel

} // namespace cuda

#endif // CUDA_API_WRAPPERS_KERNEL_HPP_
