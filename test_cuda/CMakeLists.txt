cmake_minimum_required(VERSION 3.12)
project(tcuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Check for CUDA if necessary
find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA)
  set(CUDA_FLAGS ${CUDA_LIBRARIES})
endif()

# Locate LibTorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Collect all source files
file(GLOB SOURCES "*.cpp")

add_executable(tcuda ${SOURCES})
# Link against LibTorch
target_link_libraries(tcuda ${TORCH_LIBRARIES})

if(CUDA_FOUND)
  target_link_libraries(tcuda ${CUDA_LIBRARIES})
endif()