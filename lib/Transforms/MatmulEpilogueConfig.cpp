#include "MLIRGPU/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace mlir::mlir_gpu {
namespace {

struct MatmulEpilogueConfigPass
    : public PassWrapper<MatmulEpilogueConfigPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulEpilogueConfigPass)

  void runOnOperation() final {
    getOperation().emitRemark()
        << "visited by mlir-gpu matmul epilogue config pass";
  }

  StringRef getArgument() const final {
    return "mlir-gpu-matmul-epilogue-config";
  }

  StringRef getDescription() const final {
    return "Validate and configure supported matmul epilogues";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createMatmulEpilogueConfigPass() {
  return std::make_unique<MatmulEpilogueConfigPass>();
}
void registerMLIRGPUPasses() { PassRegistration<MatmulEpilogueConfigPass>(); }
} // namespace mlir::mlir_gpu