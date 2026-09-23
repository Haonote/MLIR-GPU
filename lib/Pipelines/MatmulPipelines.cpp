#include "MLIRGPU/Pipelines/Pipelines.h"
#include "MLIRGPU/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::mlir_gpu {

void buildMatmulConfigPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createMatmulEpilogueConfigPass());
}

void registerMLIRGPUPipelines() {
  PassPipelineRegistration<>(
      "mlir-gpu-matmul-config-pipeline",
      "Validate and configure supported linalg.matmul operations",
      [](OpPassManager &pm) { buildMatmulConfigPipeline(pm); });
}

} // namespace mlir::mlir_gpu