
#ifndef MLIR_TUTORIAL_TOY2_DIALECT_H_
#define MLIR_TUTORIAL_TOY2_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "toy/ShapeInferenceInterface.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect2.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops2.h.inc"

#endif // MLIR_TUTORIAL_TOY2_DIALECT_H_
