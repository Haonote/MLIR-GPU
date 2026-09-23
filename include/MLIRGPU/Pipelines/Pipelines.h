#ifndef MLIRGPU_PIPELINES_PIPELINES_H_
#define MLIRGPU_PIPELINES_PIPELINES_H_

#include "mlir/Pass/PassManager.h"

namespace mlir::mlir_gpu {

void buildMatmulConfigPipeline(OpPassManager &pm);
void registerMLIRGPUPipelines();

} // namespace mlir::mlir_gpu

#endif // MLIRGPU_PIPELINES_PIPELINES_H_