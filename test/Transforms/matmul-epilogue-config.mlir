// RUN: mlir-gpu-opt --mlir-gpu-matmul-epilogue-config %s 2>&1 | FileCheck %s

// CHECK: remark: found linalg.matmul with M=64 N=64 K=64
// CHECK-NEXT: {{.*}}linalg.matmul

//CHECK: remark: unsupported matmul shape; expected static ranked 2D tensor operands
// CHECK-NEXT: {{.*}}linalg.matmul

// CHECK-NOT: remark:

func.func @matmul(
    %lhs: tensor<64x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

func.func @dynamic_matmul(
    %lhs: tensor<?x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<?x64xf32>) -> tensor<?x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<?x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %result : tensor<?x64xf32>
}

func.func @add(%lhs: i32, %rhs: i32) -> i32 {
  %result = arith.addi %lhs, %rhs : i32
  return %result : i32
}