#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

tflite::OpResolver& getOptimizedMicroOpResolver();

tflite::OpResolver& getFullMicroOpResolver();