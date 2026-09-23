get_property(dialect_libs 
             GLOBAL 
             PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(mlir-gpu-opt 
    mlir-gpu-opt.cpp
)

# 链接 MLIR 官方提供的核心驱动库和所有 Dialect / Pass 库
target_link_libraries(mlir-gpu-opt
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIRPass
    MLIRTransforms
    MLIRAnalysis
    MLIRIR
    MLIRSupport
    MLIRGPUCustomTransforms
    MLIRGPUCustomPipelines
)