#ifndef MLIRGPU_TRANSFORMS_PASSES_H_
#define MLIRGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::mlir_gpu {

std::unique_ptr<mlir::Pass> createMatmulEpilogueConfigPass();

void registerMLIRGPUPasses();
} // namespace mlir::mlir_gpu

#endif // MLIRGPU_TRANSFORMS_PASSES_H_