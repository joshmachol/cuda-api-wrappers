/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Execution control
 *
 * but excluding the part of this module dealing with parameter buffers:
 *
 *   cudaGetParameterBuffer
 *   cudaGetParameterBufferV2
 *   cudaSetDoubleForDevice
 *   cudaSetDoubleForHost
 *
 */
#include "cuda/api_wrappers.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <cassert>

[[noreturn]] void die(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

__global__ void foo(int bar)
{
	if (threadIdx.x == 0) {
		printf("Block %u is executing (with v = %d)\n", blockIdx.x, bar);
	}
}

std::ostream& operator<<(std::ostream& os, cuda::device::compute_capability_t cc)
{
	return os << cc.major << '.' << cc.minor;
}

int main(int argc, char **argv)
{
	const auto kernel = foo;
	const auto kernel_name = "foo"; // no reflection, sadly...

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	auto device = cuda::device::get(device_id);
	cuda::device_function_t device_function(kernel);

	// ------------------------------------------
	//  Attributes without a specific API call
	// ------------------------------------------

	auto attributes = device_function.attributes();
	std::cout
		<< "The PTX version used in compiling device function " << kernel_name
		<< " is " << attributes.ptx_version() << ".\n";

	std::string cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
	};

	// --------------------------------------------------------------
	//  Attributes with a specific API call:
	//  L1/shared memory size preference and shared memory bank size
	// --------------------------------------------------------------

	device_function.cache_preference(
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory);

	device_function.shared_memory_bank_size(
		cuda::multiprocessor_shared_memory_bank_size_option_t::four_bytes_per_bank);

	// You may be wondering why we're only setting these "attributes' but not
	// obtaining their existing values. Well - we can't! The runtime doesn't expose
	// API calls for that (as of CUDA v8.0).

	// ------------------
	//  Kernel launching
	// ------------------

	const int bar = 123;
	const unsigned num_blocks = 3;
	auto launch_config = cuda::make_launch_config(num_blocks, attributes.maxThreadsPerBlock);
	cuda::device::current::set(device_id);
	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using cuda::launch()\n" << std::flush;
	cuda::launch(kernel, launch_config, bar);
	cuda::device::current::get().synchronize();

	// but there's more than one way to launch! we can also do
	// it via the device proxy, using the default stream:

	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using device.launch()\n" << std::flush;
	device.launch(kernel, launch_config, bar);
	device.synchronize();

	// or via a stream:

	auto stream_id = device.create_stream(
		cuda::stream::no_implicit_synchronization_with_default_stream);
	auto stream = cuda::stream::make_proxy(device.id(), stream_id);
	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using stream.launch()\n" << std::flush;
	stream.launch(kernel, launch_config, bar);
	stream.synchronize();

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
