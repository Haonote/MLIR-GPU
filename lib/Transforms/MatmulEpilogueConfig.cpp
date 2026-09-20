#include "MLIRGPU/Support/TargetConfig.h"
#include "MLIRGPU/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include <memory>

namespace mlir::mlir_gpu {
namespace {

struct MatmulEpilogueConfigPass
    : public PassWrapper<MatmulEpilogueConfigPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulEpilogueConfigPass)

  void runOnOperation() final {
    auto funcOp = getOperation();
    MatmulShapeConstraints constraints;
    funcOp.walk([&constraints](Operation *op) {
      auto matmul = llvm::dyn_cast<linalg::MatmulOp>(op);
      if (!matmul)
        return;
      auto lhsType =
          llvm::dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
      auto rhsType =
          llvm::dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
      auto initType =
          llvm::dyn_cast<RankedTensorType>(matmul.getOutputs()[0].getType());

      if (!lhsType || !rhsType || !initType || lhsType.getRank() != 2 ||
          rhsType.getRank() != 2 || initType.getRank() != 2) {
        matmul.emitRemark()
            << "unsupported matmul operand types; expected ranked 2D tensors";
        return;
      }

      if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
          !initType.hasStaticShape()) {
        matmul.emitRemark() << "unsupported matmul shape; expected static "
                               "ranked 2D tensor operands";
        return;
      }

    
      int64_t M = lhsType.getDimSize(0);
      int64_t K = lhsType.getDimSize(1);
      int64_t N = rhsType.getDimSize(1);
      if (failed(constraints.isSatisfied(M, N, K))) {
        matmul.emitRemark()
            << "unsupported matmul shape; expected static positive M, N, and K";
        return;
      }

      matmul.emitRemark() << "found linalg.matmul with M=" << M << " N=" << N
                          << " K=" << K;
    });
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