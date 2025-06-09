add_mlir_dialect(StandaloneOps standalone)
add_mlir_doc(StandaloneDialect StandaloneDialect Dialect/ -gen-dialect-doc)
add_mlir_doc(StandaloneOps StandaloneOps Dialect/ -gen-op-doc)
add_dependencies(mlir-headers MLIRStandaloneOpsIncGen)
