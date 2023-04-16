# machinelearninglearning

Learning about Machine Learning.

`main.py` follows the [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) online book by Michael Nielsen.

`main_tf.py` is a tensorflow project that produced the model in `Tensorflow_models` directory. The model has a $98.81$% accuracy on the MNIST 0-9 number data set.

I use a nix flake because it is super easy to manage dependencies, but the project really only depends on `tensorflow` and `pillow` which can be installed with pip. In order to use the flake, install `nix` and simply run `nix develop` to open a shell with the correct libraries.

Code is licensed under the Feel Free to Screenshot licence (FFSL) aka do whatever you want with it.
