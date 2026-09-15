#include "MLIRGPU/Support/TargetConfig.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>

namespace {

int check(llvm::StringRef caseName, mlir::LogicalResult actual,
          bool expectSuccess) {
  if (mlir::succeeded(actual) == expectSuccess)
    return 0;

  llvm::errs() << "FAILED: " << caseName << " (expected "
               << (expectSuccess ? "success" : "failure") << ", got "
               << (mlir::succeeded(actual) ? "success" : "failure") << ")\n";
  return 1;
}

} // namespace

int main() {
  using mlir::mlir_gpu::MatmulShapeConstraints;

  int failures = 0;
  MatmulShapeConstraints defaults;

  failures +=
      check("default accepts 64x64x64", defaults.isSatisfied(64, 64, 64), true);

  failures +=
      check("dynamic M",
            defaults.isSatisfied(mlir::ShapedType::kDynamic, 64, 64), false);
  failures +=
      check("dynamic N",
            defaults.isSatisfied(64, mlir::ShapedType::kDynamic, 64), false);
  failures +=
      check("dynamic K",
            defaults.isSatisfied(64, 64, mlir::ShapedType::kDynamic), false);

  failures += check("zero M", defaults.isSatisfied(0, 64, 64), false);
  failures += check("zero N", defaults.isSatisfied(64, 0, 64), false);
  failures += check("zero K", defaults.isSatisfied(64, 64, 0), false);
  failures += check("negative M", defaults.isSatisfied(-2, 64, 64), false);
  failures += check("negative N", defaults.isSatisfied(64, -2, 64), false);
  failures += check("negative K", defaults.isSatisfied(64, 64, -2), false);

  const std::array<int64_t, 2> tooShort = {64, 64};
  const std::array<int64_t, 3> validDims = {64, 64, 64};
  const std::array<int64_t, 4> tooLong = {64, 64, 64, 64};
  failures +=
      check("ArrayRef has two dimensions",
            defaults.isSatisfied(llvm::ArrayRef<int64_t>(tooShort)), false);
  failures +=
      check("ArrayRef has three valid dimensions",
            defaults.isSatisfied(llvm::ArrayRef<int64_t>(validDims)), true);
  failures +=
      check("ArrayRef has four dimensions",
            defaults.isSatisfied(llvm::ArrayRef<int64_t>(tooLong)), false);

  MatmulShapeConstraints customMin;
  customMin.minM = 32;
  customMin.minN = 64;
  customMin.minK = 16;
  failures +=
      check("custom minimum boundary", customMin.isSatisfied(32, 64, 16), true);
  failures +=
      check("below custom minimum M", customMin.isSatisfied(31, 64, 16), false);
  failures +=
      check("below custom minimum N", customMin.isSatisfied(32, 63, 16), false);
  failures +=
      check("below custom minimum K", customMin.isSatisfied(32, 64, 15), false);

  MatmulShapeConstraints customAlignment;
  customAlignment.alignmentM = 16;
  customAlignment.alignmentN = 16;
  customAlignment.alignmentK = 8;
  failures += check("custom alignment accepts aligned input",
                    customAlignment.isSatisfied(32, 48, 24), true);
  failures += check("custom alignment rejects M",
                    customAlignment.isSatisfied(33, 48, 24), false);
  failures += check("custom alignment rejects N",
                    customAlignment.isSatisfied(32, 49, 24), false);
  failures += check("custom alignment rejects K",
                    customAlignment.isSatisfied(32, 48, 25), false);

  auto checkInvalidConfiguration = [&](llvm::StringRef caseName,
                                       MatmulShapeConstraints constraints) {
    failures += check(caseName, constraints.isSatisfied(64, 64, 64), false);
  };

  MatmulShapeConstraints invalid;
  invalid.minM = 0;
  checkInvalidConfiguration("zero minM", invalid);
  invalid = defaults;
  invalid.minM = -1;
  checkInvalidConfiguration("negative minM", invalid);
  invalid = defaults;
  invalid.minN = 0;
  checkInvalidConfiguration("zero minN", invalid);
  invalid = defaults;
  invalid.minN = -1;
  checkInvalidConfiguration("negative minN", invalid);
  invalid = defaults;
  invalid.minK = 0;
  checkInvalidConfiguration("zero minK", invalid);
  invalid = defaults;
  invalid.minK = -1;
  checkInvalidConfiguration("negative minK", invalid);
  invalid = defaults;
  invalid.alignmentM = 0;
  checkInvalidConfiguration("zero alignmentM", invalid);
  invalid = defaults;
  invalid.alignmentM = -1;
  checkInvalidConfiguration("negative alignmentM", invalid);
  invalid = defaults;
  invalid.alignmentN = 0;
  checkInvalidConfiguration("zero alignmentN", invalid);
  invalid = defaults;
  invalid.alignmentN = -1;
  checkInvalidConfiguration("negative alignmentN", invalid);
  invalid = defaults;
  invalid.alignmentK = 0;
  checkInvalidConfiguration("zero alignmentK", invalid);
  invalid = defaults;
  invalid.alignmentK = -1;
  checkInvalidConfiguration("negative alignmentK", invalid);

  if (failures != 0)
    llvm::errs() << failures << " TargetConfig test case(s) failed\n";
  return failures == 0 ? 0 : 1;
}
