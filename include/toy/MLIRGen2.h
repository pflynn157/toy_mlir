
#ifndef TOY2_MLIRGEN_H
#define TOY2_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace toy {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen2(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
} // namespace toy

#endif // TOY2_MLIRGEN_H
