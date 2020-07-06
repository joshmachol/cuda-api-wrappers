/**
 * @file miscellany.hpp
 *
 * @brief Miscellaneous functionality which does not fit in another file,
 * and does not depend on the main proxy classes
 *
 */
#ifndef CUDA_API_WRAPPERS_MISCELLANY_HPP_
#define CUDA_API_WRAPPERS_MISCELLANY_HPP_

#include <cuda/api/types.hpp>

#include <cuda_runtime_api.h>
#include <cuda/api/error.hpp>

#include <cuda.h>
#include <ostream>
#include <utility>

namespace cuda {

/**
 * Obtains the CUDA Runtime version
 *
 * @note unlike {@ref maximum_supported_by_driver()}, 0 cannot be returned,
 * as we are actually using the runtime to obtain the version, so it does
 * have _some_ version.
 */
inline void initialize_driver() {
	constexpr const unsigned dummy_flags { 0 }; // this is the only allowed value for flags
	auto status = cuInit(dummy_flags);
	throw_if_error(status, "Failed initializing the CUDA driver");
}

namespace device {

/**
 * Get the number of CUDA devices usable on the system (with the current CUDA
 * library and kernel driver)
 *
 * @note This _should_ be returning an unsigned value; unfortunately, device::handle_t  is
 * signed in CUDA for some reason and we maintain compatibility (although this might
 * change in the future). So... the returned type is the same as in cudaGetDeviceCount,
 * a signed integer.
 *
 * @return the number of CUDA devices on this system
 * @throws cuda::error if the device count could not be obtained
 */
inline device::id_t  count()
{
	int device_count = 0; // Initializing, just to be on the safe side
	status_t result = cudaGetDeviceCount(&device_count);
	if (result == status::no_device) {
		return 0;
	}
	else {
		throw_if_error(result, "Failed obtaining the number of CUDA devices on the system");
	}
	if (device_count < 0) {
		throw ::std::logic_error("cudaGetDeviceCount() reports an invalid number of CUDA devices");
	}
	return device_count;
}

} // namespace device


} // namespace cuda

#endif // CUDA_API_WRAPPERS_MISCELLANY_HPP_
