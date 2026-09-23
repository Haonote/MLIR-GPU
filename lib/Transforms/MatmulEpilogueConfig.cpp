#include "MLIRGPU/Support/TargetConfig.h"
#include "MLIRGPU/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <memory>

namespace mlir::mlir_gpu {
namespace {

struct MatmulEpilogueConfigPass
    : public PassWrapper<MatmulEpilogueConfigPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulEpilogueConfigPass)

  void runOnOperation() final {
    auto funcOp = getOperation();
    MatmulTargetConfig targetConfig;
    funcOp.walk([&targetConfig](Operation *op) {
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

      if (!llvm::isa<Float32Type>(lhsType.getElementType()) ||
          !llvm::isa<Float32Type>(rhsType.getElementType()) ||
          !llvm::isa<Float32Type>(initType.getElementType())) {
        matmul.emitRemark() << "unsupported matmul element types; expected f32";
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
      if (failed(targetConfig.shapeConstraints.isSatisfied(M, N, K))) {
        matmul.emitRemark()
            << "unsupported matmul shape; expected static positive M, N, and K";
        return;
      }
      auto indexingMaps = matmul.getIndexingMapsArray();

      MLIRContext *context = matmul.getContext();
      AffineExpr d0 = getAffineDimExpr(0, context);
      AffineExpr d1 = getAffineDimExpr(1, context);
      AffineExpr d2 = getAffineDimExpr(2, context);

      SmallVector<AffineMap> expectedIndexingMaps = {
          AffineMap::get(3, 0, {d0, d2}, context),
          AffineMap::get(3, 0, {d2, d1}, context),
          AffineMap::get(3, 0, {d0, d1}, context),
      };

      if (indexingMaps.size() != expectedIndexingMaps.size() ||
          !std::equal(indexingMaps.begin(), indexingMaps.end(),
                      expectedIndexingMaps.begin())) {
        matmul.emitRemark() << "unsupported matmul layout; expected standard "
                               "MxK * KxN indexing";
        return;
      }

      const auto tileSizes = targetConfig.selectTileSizes(M, N, K);
      matmul->setAttr(
          "mlir_gpu.tile_sizes",
          DenseI64ArrayAttr::get(matmul.getContext(),
                                 {tileSizes.m, tileSizes.n, tileSizes.k}));
    });
  }

  StringRef getArgument() const final {
    return "mlir-gpu-matmul-epilogue-config";
  }

  StringRef getDescription() const final {
    return "Validate and configure supported linalg.matmul operations";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createMatmulEpilogueConfigPass() {
  return std::make_unique<MatmulEpilogueConfigPass>();
}
void registerMLIRGPUPasses() { PassRegistration<MatmulEpilogueConfigPass>(); }
} // namespace mlir::mlir_gpu