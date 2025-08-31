cmake_minimum_required(VERSION 3.18)
project(yolo_trt LANGUAGES CXX)

# Find CUDA and OpenCV
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# ---------------------------
# TensorRT paths (adjust if needed)
# ---------------------------
set(TENSORRT_INCLUDE_DIR /path-to-TensorRT-include)
set(TENSORRT_LIB_DIR     /path-to-TensorRT-lib)
set(TENSORRT_LIB         ${TENSORRT_LIB_DIR}/libnvinfer.so)
set(TENSORRT_PLUGIN_LIB  ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so)

# ---------------------------
# Include directories
# ---------------------------
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# ---------------------------
# Library directories
# ---------------------------
link_directories(
    ${CUDAToolkit_LIBRARY_DIRS}
    ${TENSORRT_LIB_DIR}
)

# ---------------------------
# Build executable
# ---------------------------
add_executable(yolo_trt main.cpp)

# ---------------------------
# Link libraries
# ---------------------------
target_link_libraries(yolo_trt
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
)


