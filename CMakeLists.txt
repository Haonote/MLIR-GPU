cmake_minimum_required(VERSION 3.20)
project(MLIR-GPU LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找 MLIR 包
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# 2. 引入 MLIR & LLVM 的 CMake 模块路径
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 3. 设置头文件搜索路径
include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# 4. 添加子目录
add_subdirectory(tools/mlir-gpu-opt)

# 5. 测试配置
find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit
    PATHS
    ${LLVM_TOOLS_BINARY_DIR}
    /home/hao/miniconda3/bin
    /home/hao/project/LLVM/llvm-project/build/bin
)
add_subdirectory(test)
add_subdirectory(lib)
