# SBPNN
## Usage
- run make.
- copy the trainning file and testing file to ./bin/
- Choose whether to test the existing weights file or to train one. The default neural network has 24 * 24 inputs and 64 hidden units and 10 outputs. To change this, please do it in main.cpp.
- Also, the default training acceptable error is 400. This means from all of your training case, if the total error <= 400, the training is successful and the weights file will be saved. Otherwise the training will be continued.
- In case the training speed is too slow and the error can't reach expecting value. I used **thread** in cbpnn_main.cpp, if you think the error is low enough and you don't want to wait, you can type in *save* to force save the existing weights.
## TODO
* Accept command line options.
* Recfactor SBPNN to make it more readable.

## Contact
For further information, please create in the issues.
