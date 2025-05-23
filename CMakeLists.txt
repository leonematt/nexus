cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(nexus CXX C)
#include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(NEXUS_BUILD_PYTHON_MODULE "Build Python Nexus bindings" OFF)
option(NEXUS_ENABLE_LOGGING "Enable Nexus Logging" ON)
option(NEXUS_BUILD_PLUGINS "Build the runtime plugins" ON)
#set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "Enable different codegen backends")

# Ensure Python3 vars are set correctly
# used conditionally in this file and by lit tests

# Customized release build type with assertions: TritonRelBuildWithAsserts
if(NOT MSVC)
  set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
  set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
  set(CMAKE_C_FLAGS_TRITONBUILDWITHO1 "-O1")
  set(CMAKE_CXX_FLAGS_TRITONBUILDWITHO1 "-O1")
else()
  set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "/Zi /RTC1 /bigobj /Zc:preprocessor /permissive-")
  set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "/Zi /RTC1 /bigobj /Zc:preprocessor /permissive-")
  set(CMAKE_EXE_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
  set(CMAKE_MODULE_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
  set(CMAKE_SHARED_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(NOT WIN32)
  find_library(TERMINFO_LIBRARY tinfo)
endif()

#if(TRITON_BUILD_UT)
#  # This is an aggregate target for all unit tests.
#  add_custom_target(TritonUnitTests)
#  set_target_properties(TritonUnitTests PROPERTIES FOLDER "Triton/Tests")
#  include(AddTritonUnitTest)
#endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")
endif()


# Disable warnings that show up in external code (gtest;pybind11)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4624 /wd4715 /wd4530")
endif()

if(NEXUS_ENABLE_LOGGING)
  add_compile_definitions(NEXUS_LOGGING)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

# link_directories(${LLVM_LIBRARY_DIR})
#add_subdirectory(include)

# TODO: Figure out which target is sufficient to fix errors; triton is
# apparently not enough. Currently set linking libstdc++fs for all targets
# to support some old version GCC compilers like 8.3.0.
if (NOT WIN32 AND NOT APPLE AND NOT BSD)
  link_libraries(stdc++fs)
endif()


# -----

# ------
if(NEXUS_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  # Python Interpreter is used to run lit tests
  find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

  #get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
  #get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)
  #set(TRITON_LIBRARIES
  #  ${triton_libs}
  #  ${triton_plugins}

  #  # mlir
  #  MLIRAMDGPUDialect
  #  MLIRNVVMDialect
  #  MLIRNVVMToLLVMIRTranslation
  #  MLIRGPUToNVVMTransforms
  #  MLIRGPUToGPURuntimeTransforms
  #  MLIRGPUTransforms
  #  MLIRIR
  #  MLIRControlFlowToLLVM
  #  MLIRBytecodeWriter
  #  MLIRPass
  #  MLIRTransforms
  #  MLIRLLVMDialect
  #  MLIRSupport
  #  MLIRTargetLLVMIRExport
  #  MLIRMathToLLVM
  #  MLIRROCDLToLLVMIRTranslation
  #  MLIRGPUDialect
  #  MLIRSCFToControlFlow
  #  MLIRIndexToLLVM
  #  MLIRGPUToROCDLTransforms
  #  MLIRUBToLLVM

  #  # LLVM
  #  LLVMPasses
  #  LLVMNVPTXCodeGen
  #  # LLVMNVPTXAsmPrinter
  #  LLVMAMDGPUCodeGen
  #  LLVMAMDGPUAsmParser
  #  LLVMRISCVCodeGen
  #  LLVMRISCVAsmParser

  #  Python3::Module
  #  pybind11::headers

  #)

  #set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
  #add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})
  #add_library(nexus SHARED ${PYTHON_SRC_PATH}/main.cc
  #                ${PYTHON_SRC_PATH}/ir.cc
  #                ${PYTHON_SRC_PATH}/passes.cc
  #                ${PYTHON_SRC_PATH}/interpreter.cc
  #                ${PYTHON_SRC_PATH}/llvm.cc)

  ## Link triton with its dependencies
  #target_link_libraries(triton PRIVATE ${TRITON_LIBRARIES})
  #if(WIN32)
  #  target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
  #  set_target_properties(triton PROPERTIES SUFFIX ".pyd")
  #  set_target_properties(triton PROPERTIES PREFIX "lib")
  #else()
  #  target_link_libraries(triton PRIVATE z)
  #endif()
  #target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
endif()

if (UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

if(NEXUS_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

  # Check if the platform is MacOS
  if(APPLE)
    set(PYTHON_LDFLAGS "-undefined dynamic_lookup")
  endif()

  target_link_options(nexus PRIVATE ${PYTHON_LDFLAGS})
endif()

#find_package(Threads REQUIRED)

add_subdirectory(third_party/json json)
#@@@ include_directories(third_party/json/include)
add_subdirectory(src)
#add_subdirectory(test)

if(NEXUS_BUILD_PLUGINS)
  add_subdirectory(plugins)
endif()

#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#  # This target runs all the unit tests.
#  add_custom_target(check-triton-unit-tests
#    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
#    DEPENDS TritonUnitTests
#    USES_TERMINAL
#  )
#endif()
