#set(LLVM_TARGET_DEFINITIONS Ops.td)
#mlir_tablegen(Ops.h.inc -gen-op-decls)
#mlir_tablegen(Ops.cpp.inc -gen-op-defs)
#mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
#mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
#add_public_tablegen_target(ToyDialect)


# Most dialects should use add_mlir_dialect().  See examples/standalone.
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyDialect1)

set(LLVM_TARGET_DEFINITIONS Ops2.td)
mlir_tablegen(Ops2.h.inc -gen-op-decls)
mlir_tablegen(Ops2.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect2.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect2.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToySubDialect1)

# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyDialect2)
