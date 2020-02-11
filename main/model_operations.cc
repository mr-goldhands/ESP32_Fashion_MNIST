#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#define OPERATIONS_QUANTITY 1U

// Pull in only the operation implementations we need.
// This relies on a complete list of all the ops needed by this graph.
// An easier approach is to just use the AllOpsResolver, but this will
// incur some penalty in code space for op implementations that are not
// needed by this graph.
// To get list of operations in graph run interpreter.get_tensor_details() in python with TF Lite model interpreter
tflite::OpResolver& getOptimizedMicroOpResolver() {
    static tflite::MicroOpResolver<OPERATIONS_QUANTITY> micro_op_resolver;
 
    // AddBuiltin(<operator ID>, <registration>, [min version], [max version])

    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::ops::micro::Register_FULLY_CONNECTED(), 1, 4);
    
    // micro_op_resolver.AddBuiltin(
    //     tflite::BuiltinOperator_MAX_POOL_2D,
    //     tflite::ops::micro::Register_MAX_POOL_2D());
    
    // micro_op_resolver.AddBuiltin(
    //     tflite::BuiltinOperator_CONV_2D,
    //     tflite::ops::micro::Register_CONV_2D(), 1, 3);
    
    // micro_op_resolver.AddBuiltin(
    //     tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
    //     tflite::ops::micro::Register_DEPTHWISE_CONV_2D(), 1, 3);
    
    // micro_op_resolver.AddBuiltin(
    //     tflite::BuiltinOperator_RELU,
    //     tflite::ops::micro::Register_RELU());

    return micro_op_resolver;
}

tflite::OpResolver& getFullMicroOpResolver() {
    static tflite::ops::micro::AllOpsResolver micro_op_resolver;
    return micro_op_resolver;
}
