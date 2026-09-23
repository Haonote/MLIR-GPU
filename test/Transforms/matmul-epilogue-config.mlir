// RUN: mlir-gpu-opt --mlir-gpu-matmul-epilogue-config %s 2>&1 | FileCheck %s
// RUN: mlir-gpu-opt --mlir-gpu-matmul-config-pipeline %s 2>&1 | FileCheck %s

/// CHECK: remark: unsupported matmul element types; expected f32
// CHECK-NEXT: {{.*}}linalg.matmul

// CHECK: remark: unsupported matmul shape; expected static ranked 2D tensor operands
// CHECK-NEXT: {{.*}}linalg.matmul

// CHECK: remark: unsupported matmul shape; expected static positive M, N, and K
// CHECK-NEXT: {{.*}}linalg.matmul

// CHECK: remark: unsupported matmul layout; expected standard MxK * KxN indexing
// CHECK-NEXT: {{.*}}linalg.matmul

// CHECK-LABEL: func.func @matmul(
// CHECK: linalg.matmul {mlir_gpu.tile_sizes = array<i64: 16, 16, 16>}

// CHECK-LABEL: func.func @large_matmul(
// CHECK: linalg.matmul {mlir_gpu.tile_sizes = array<i64: 32, 32, 16>}

// CHECK-LABEL: func.func @transposed_lhs(
// CHECK-NOT: mlir_gpu.tile_sizes
// CHECK: return

// CHECK-NOT: remark:

func.func @f16_matmul(
    %lhs: tensor<64x64xf16>,
    %rhs: tensor<64x64xf16>,
    %init: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %result : tensor<64x64xf16>
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

func.func @zero_m(
    %lhs: tensor<0x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<0x64xf32>) -> tensor<0x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<0x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<0x64xf32>) -> tensor<0x64xf32>
  return %result : tensor<0x64xf32>
}

func.func @matmul(
    %lhs: tensor<64x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

func.func @large_matmul(
    %lhs: tensor<128x64xf32>,
    %rhs: tensor<64x128xf32>,
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}

func.func @transposed_lhs(
    %lhs: tensor<64x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul
      indexing_maps = [
        affine_map<(m, n, k) -> (k, m)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>]
      ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

func.func @add(%lhs: i32, %rhs: i32) -> i32 {
  %result = arith.addi %lhs, %rhs : i32
  return %result : i32
}