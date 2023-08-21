cmake_minimum_required(VERSION 3.10)
project(cgotorch)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Check for CUDA if necessary
find_package(CUDA QUIET)
if(CUDA_FOUND)
  add_definitions(-DUSE_CUDA)
  set(CUDA_FLAGS ${CUDA_LIBRARIES})
endif()

# Locate LibTorch
find_package(Torch REQUIRED)

# Collect all source files
file(GLOB SOURCES "*.cc")

# Define the shared library
add_library(cgotorch SHARED ${SOURCES})

# Linking
target_link_libraries(cgotorch PUBLIC "${TORCH_LIBRARIES}")

# Include directories
target_include_directories(cgotorch PUBLIC ${TORCH_INCLUDE_DIRS})
