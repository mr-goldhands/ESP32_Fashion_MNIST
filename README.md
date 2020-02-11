# Fashion MNIST for ESP32 example

This example shows how you can use Tensorflow Lite for Microcontrollers to run a  neural network to classify images of clothing.  It is designed to
run on ESP32 platform.

Here ESP32 was used as standalone computational device.

### Hardware

![LILYGO® TTGO T-Camera ESP32 WROVER](https://github.com/mr-goldhands/ESP32_Fashion_MNIST/blob/master/docs/hardware/esp32_both_sides.jpg)

It is [LILYGO® TTGO T-Camera ESP32 WROVER](http://www.lilygo.cn/prod_view.aspx?TypeId=50030&Id=1122&FId=t3:50030:3) that was [bought here](https://www.aliexpress.com/item/32968683765.html).

### Software. The idea.

Sometimes it makes sense using more than one MCU in the IoT project. In this project, I decided to consider ESP32 MCU as a standalone computational device. In my case, I did not want to dig in ESP32 camera API and wanted to be focused on ML functionality. I wanted just only send data from my PC to the microcontroller to feed NN and then receive results back. Almost all microcontrollers have at least one UART controller which makes such an approach quite universal (at least for testing your NN on certain hardware).

I used the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset provided by [TensorFlow Datasets](https://www.tensorflow.org/datasets) to build model with input images 14x14 pixels size. I deployed the model with TensorFlow Lite for Microcontrollers to ESP32. Then I run [Test_TFLite_Micro.ipynb](https://github.com/mr-goldhands/ESP32_Fashion_MNIST/blob/master/python/Test_TFLite_Micro.ipynb) to send test data to MCU and receive predictions back to PC.

![PC will send data to the microcontroller to run inference on the NN](https://github.com/mr-goldhands/ESP32_Fashion_MNIST/blob/master/docs/Project_diagram.jpg)

### [Please read the article on Medium about the development process in details](https://medium.com/@dmytro.korablyov/first-steps-with-esp32-and-tensorflow-lite-for-microcontrollers-c2d8e238accf)
