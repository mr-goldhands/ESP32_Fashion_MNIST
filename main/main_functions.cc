#include "driver/uart.h"
#include "driver/gpio.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "main_functions.h"
#include "model_data.h"

void initUart(uart_port_t uart_num)
{
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB};

    // We will not use a buffer for sending data.
    uart_driver_install(uart_num, RX_BUF_SIZE * 2, 0, 0, NULL, 0);

    // Configure UART parameters
    ESP_ERROR_CHECK(uart_param_config(uart_num, &uart_config));
    // Cnfigure the physical GPIO pins to which the UART device will be connected.
    ESP_ERROR_CHECK(uart_set_pin(uart_num, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

void readUartBytes(float *data, int imageSize)
{
    uint8_t *rxBuffer = (uint8_t *)malloc(RX_BUF_SIZE + 1);
    int rxIdx = 0;
    int rxBytes = 0;

    for (;;)
    {
        rxBytes = uart_read_bytes(UART_NUMBER, rxBuffer, RX_BUF_SIZE, 1000 / portTICK_RATE_MS);
        if (rxBytes > 0)
        {
            for (int i = 0; i < rxBytes; rxIdx++, i++)
            {
                data[rxIdx] = static_cast<float>(rxBuffer[i]) / 255.0f;
            }
        }
        if (rxIdx >= imageSize - 1)
        {
            rxIdx = 0;
            break;
        }
    }
    return;
}

int sendData(const char *data)
{
    const int len = strlen(data);
    const int txBytes = uart_write_bytes(UART_NUMBER, data, len);
    return txBytes;
}

void normalizeImageData(float *data, int imageSize)
{
    for (int i = 0; i < imageSize; i++)
    {
        data[i] = data[i] / 255.0f;
    }
}

void sendBackPredictions(TfLiteTensor *output)
{
    // Read the predicted y values from the model's output tensor
    char str[250] = {0};
    char buf[20] = {0};
    int numElements = output->dims->data[1];
    for (int i = 0; i < numElements; i++)
    {
        sprintf(buf, "%e,", static_cast<float>(output->data.f[i]));
        strcat(str, buf);
    }
    strcat(str, "\n");
    sendData(str);
}
