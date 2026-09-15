#ifndef MLIRGPU_SUPPORT_TARGETCONFIG_H_
#define MLIRGPU_SUPPORT_TARGETCONFIG_H_

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace mlir::mlir_gpu {

struct MatmulShapeConstraints {
  int64_t minM = 1;
  int64_t minN = 1;
  int64_t minK = 1;
  int64_t alignmentM = 1;
  int64_t alignmentN = 1;
  int64_t alignmentK = 1;

  mlir::LogicalResult isSatisfied(llvm::ArrayRef<int64_t> dims) const {
    if (dims.size() != 3)
      return mlir::failure();

    int64_t m = dims[0];
    int64_t n = dims[1];
    int64_t k = dims[2];

    return isSatisfied(m, n, k);
  }

  mlir::LogicalResult isSatisfied(int64_t m, int64_t n, int64_t k) const {
    if (minM <= 0 || minN <= 0 || minK <= 0 || alignmentM <= 0 ||
        alignmentN <= 0 || alignmentK <= 0)
      return mlir::failure();

    auto isValidDim = [](int64_t val, int64_t minimum, int64_t alignment) {
      if (mlir::ShapedType::isDynamic(val))
        return false;
      return val >= minimum && val % alignment == 0;
    };

    if (isValidDim(m, minM, alignmentM) && isValidDim(n, minN, alignmentN) &&
        isValidDim(k, minK, alignmentK))
      return mlir::success();

    return mlir::failure();
  }
};

} // namespace mlir::mlir_gpu

#endif // MLIRGPU_SUPPORT_TARGETCONFIG_H_
