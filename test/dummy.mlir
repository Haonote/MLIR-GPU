// RUN: mlir-gpu-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_add
// CHECK: %[[RES:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK: return %[[RES]] : i32
func.func @test_add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}