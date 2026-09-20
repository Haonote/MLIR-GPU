add_mlir_library(MLIRGPUCustomTransforms
  MatmulEpilogueConfig.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
)