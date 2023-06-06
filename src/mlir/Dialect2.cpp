#include "toy/Dialect2.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::toy2;

#include "toy/Dialect2.cpp.inc"

//===----------------------------------------------------------------------===//
// ToyDialect2
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect2::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops2.cpp.inc"
      >();
  //addInterfaces<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops2.cpp.inc"
