cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX)
  # Upstream's bundled spdlog/fmt source requires the MSVC UTF-8 source and execution charset.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  # C++ exceptions must be enabled for host .cpp files. CUDA passes /EHsc via
  # nvcc -Xcompiler, but plain cl.exe needs it here too; without it every
  # throw/try/catch (JSON parse rejection, invalid context-cost provenance,
  # and so on) compiles into a broken unwind that ends in an uncatchable
  # FAST_FAIL_GS_COOKIE_INITIALIZATION_FAILURE at run time.
  if(NOT CMAKE_CXX_FLAGS MATCHES "(^|[ -/])EHsc([ ;]|$)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif()
  # Ship symbolizable Release executables.
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    foreach(_ninfer_rel_flag "/O2" "/Ob2" "/DNDEBUG" "/Zi")
      if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^|[ -/])${_ninfer_rel_flag}([ ;]|$)")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${_ninfer_rel_flag}")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${_ninfer_rel_flag}")
      endif()
    endforeach()
    unset(_ninfer_rel_flag)
    if(NOT CMAKE_EXE_LINKER_FLAGS_RELEASE MATCHES "/DEBUG")
      set(CMAKE_EXE_LINKER_FLAGS_RELEASE
          "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
      set(CMAKE_SHARED_LINKER_FLAGS_RELEASE
          "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG")
    endif()
  endif()
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
