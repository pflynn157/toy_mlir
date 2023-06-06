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

//===----------------------------------------------------------------------===//
// FunctOp
//===----------------------------------------------------------------------===//

def FunctOp : Toy_Op2<"func", [
    DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface,
    IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "toy.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// RetOp
//===----------------------------------------------------------------------===//

def RetOp : Toy_Op2<"return", [Pure, HasParent<"FunctOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

#endif // TOY2_OPS
