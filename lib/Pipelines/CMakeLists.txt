add_mlir_library(MLIRGPUCustomPipelines
  MatmulPipelines.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRGPUCustomTransforms
)