# 替换 @变量@ 生成 build 目录下的 lit 站点配置
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_executable(TargetConfigTest
    Support/TargetConfigTest.cpp
)
llvm_update_compile_flags(TargetConfigTest)
target_link_libraries(TargetConfigTest PRIVATE
    MLIRIR
    LLVMSupport
)

# 依赖工具：mlir-gpu-opt，外部工具依赖 FileCheck
add_custom_target(check-mlir-gpu
    COMMAND $<TARGET_FILE:TargetConfigTest>
    COMMAND ${LLVM_EXTERNAL_LIT} -v ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS mlir-gpu-opt TargetConfigTest
    COMMENT "Running MLIR-GPU lit test suite..."
)

add_custom_target(check-tiny DEPENDS check-mlir-gpu)
