#ifndef TOY2_OPS
#define TOY2_OPS

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/ShapeInferenceInterface.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect2 : Dialect {
  let name = "toy2";
  let cppNamespace = "::mlir::toy2";
}

class Toy_Op2<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect2, mnemonic, traits>;

#endif // TOY2_OPS