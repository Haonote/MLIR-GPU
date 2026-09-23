#include "MLIRGPU/Transforms/Passes.h"
#include "MLIRGPU/Pipelines/Pipelines.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllPasses();
    mlir::mlir_gpu::registerMLIRGPUPasses();
    mlir::mlir_gpu::registerMLIRGPUPipelines();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MLIR-GPU optimizer driver\n", registry)
    );
}