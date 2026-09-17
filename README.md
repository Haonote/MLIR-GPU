# MLIR-GPU

学习型 MLIR out-of-tree 项目。目标是将受限的静态 FP32 `matmul + bias + ReLU` 子图编译到 NVIDIA GPU，并通过轻量 CUDA Driver Runtime 校验和分析性能。

## 当前状态（2026-09-15）

- 已完成 `mlir-gpu-opt` 工程入口、CMake、lit/FileCheck；干净项目构建目录验证通过，现有 1 个冒烟测试通过。
- `TargetConfig.h` 是尚待修正和接入的草稿；pass/pipeline 头文件与项目库尚未完成。
- 尚未支持目标 Linalg 子图、自有优化、项目 GPU 编译流水线、Runtime 或 benchmark。
- 已知问题：GPU 上游示例在当前工具中因缺少扩展接口注册崩溃；当前工具链还存在 libdevice 路径问题。


## 构建与测试

前提：已构建并安装 LLVM/MLIR，含 CMake package、所需库、FileCheck 和 lit。当前验证版本为 LLVM `24.0.0git`，revision `e046dce4a4c80610b49d67bc02c85f86b1a6353d`，targets 为 `host;NVPTX`。其他版本尚未验证。

在本仓库根目录执行，把下面两条路径替换为自己的安装路径：

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=/path/to/lit \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target check-mlir-gpu -j2
build/bin/mlir-gpu-opt test/dummy.mlir
```

当前预期为 `dummy.mlir` 1/1 通过；这个测试不需要 GPU，也不代表 GPU 编译或执行已验证。

## 计划中的 V1

输入契约与诊断 → 可消费的配置/调度 → GPU/NVVM/PTX → 明确 kernel ABI 的 Driver runner → 数值测试与同口径性能实验。

验收包括至少 8 个支持 shape、独立拒绝测试、配置引起的 IR 变化、一次有证据的优化、原始数据和复现命令。动态 shape、Tensor Core、多卡和其他后端不属于当前已实现能力。