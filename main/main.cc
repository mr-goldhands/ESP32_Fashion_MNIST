
#include "string.h"
#include <stdio.h>

#include "esp_system.h"

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "main_functions.h"

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"
#include "model_operations.h"

int totalExpectedDataAmount = INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT * INPUT_IMAGE_CHANNELS;

tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// The name of this function is important for Arduino compatibility.
void setup()
{
    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Pepare the model.
    model = tflite::GetModel(fashion_mnist_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Prepare graph model operations resolver.
    // Here I use resolver that loads all possible set of operations.
    // It will be better to use MicroOpResolver that loads
    // only really needed set of operations.
    static tflite::OpResolver &resolver = getFullMicroOpResolver();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        error_reporter->Report("AllocateTensors() failed.");
        return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void doInference()
{
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        error_reporter->Report("Invoke failed.");
        return;
    }
}

extern "C" void app_main(void)
{
    initUart(UART_NUMBER);
    setup();
    for (;;)
    {
        readUartBytes(input->data.f, totalExpectedDataAmount);
        doInference();
        sendBackPredictions(output);
    }
}