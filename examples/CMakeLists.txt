enable_language(CXX)
enable_language(CUDA)

include(FindCUDA/select_compute_arch)
# This _should_ work with CMake 3.18 and later...:
#
#if((NOT DEFINED CUDA_ARCH_LIST) OR ("${CUDA_ARCH_LIST}" STREQUAL ""))
#	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#	string(REPLACE "." "" INSTALLED_GPU_CC_LIST "${INSTALLED_GPU_CCS_3}")
#	set(CUDA_ARCH_LIST "${INSTALLED_GPU_CC_LIST}" CACHE STRING "GPU Architecture / Compute Capability codes to use; semicolon-separates" FORCE)
#
#endif()
#LIST(APPEND CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#
# ... but it doesn't, for some versions of CUDA, and on some platforms. Somehow, the script messes up the flags passed to NVCC
# So, for now, we're falling back on the old-style method of directly appending to CMAKE_CUDA_FLAGS rather than going through CUDA_ARCHITECTURES.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
	cmake_policy(SET CMP0104 OLD)
endif()
if((NOT DEFINED CUDA_ARCH_FLAGS) OR ("${CUDA_ARCH_FLAGS}" STREQUAL ""))
	cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_1 Auto)
	set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS} CACHE STRING "CUDA -gencode parameters")
	string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
else()
	set(CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

include(CompileWithWarnings)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall" )
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

link_libraries(runtime-api)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
add_executable(vectorAdd modified_cuda_samples/vectorAdd/vectorAdd.cu)
add_executable(vectorAddMapped modified_cuda_samples/vectorAddMapped/vectorAddMapped.cu)
add_executable(vectorAddManaged modified_cuda_samples/vectorAddManaged/vectorAddManaged.cu)
add_executable(inlinePTX modified_cuda_samples/inlinePTX/inlinePTX.cu)
add_executable(simpleStreams modified_cuda_samples/simpleStreams/simpleStreams.cu)
add_executable(simpleIPC modified_cuda_samples/simpleIPC/simpleIPC.cu)
add_executable(p2pBandwidthLatencyTest modified_cuda_samples/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu)
add_executable(asyncAPI modified_cuda_samples/asyncAPI/asyncAPI.cu)
#----
add_custom_target(modified_cuda_samples)
add_dependencies(modified_cuda_samples vectorAdd inlinePTX simpleStreams simpleIPC)

add_executable(version_management by_runtime_api_module/version_management.cpp)
add_executable(error_handling by_runtime_api_module/error_handling.cu)
add_executable(device_management by_runtime_api_module/device_management.cpp)
add_executable(execution_control by_runtime_api_module/execution_control.cu)

add_executable(stream_management by_runtime_api_module/stream_management.cu)
add_executable(event_management by_runtime_api_module/event_management.cu)
add_executable(unified_addressing by_runtime_api_module/unified_addressing.cpp)
add_executable(io_compute_overlap_with_streams other/io_compute_overlap_with_streams.cu)
add_executable(inclusion_in_two_translation_units other/inclusion_in_two_translation_units/main.cpp other/inclusion_in_two_translation_units/second_tu.cpp )

if(NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
	# A weird NVCC-only linking issue
	set_target_properties(execution_control PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

	# __nv_tex_surf_handler not yet implemented in Clang (as of 11.0)
	# https://reviews.llvm.org/D76365
	add_executable(array_management other/array_management.cu)
endif()
