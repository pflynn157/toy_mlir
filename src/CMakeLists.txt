# For a better template to copy, see examples/standalone
#add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(SRC
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  )

add_llvm_executable(toyc ${SRC})
add_dependencies(toyc ToyDialect)
llvm_update_compile_flags(toyc)

include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
