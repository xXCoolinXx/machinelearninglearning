import tensorflow as tf 
from tensorflow import keras
import gzip
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image
import datetime

def expand_tensor(t): #Takes a 4-Tensor 
    def roll(shift):
        return tf.roll(t, shift, axis=[1,2])

    up = roll([0,1])
    down = roll([0,-1])
    left = roll([-1,0])
    right = roll([1,0])

    return tf.concat([t,up,down,left,right], axis=0)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
    x_val = x_train[-10000:]
    x_train = x_train[:-10000]
    x_train = expand_tensor(x_train) #Inrease training data like a boss
    x_test  =  x_test.reshape(10000, 28, 28, 1).astype("float32") / 255
    
    

    y_train = to_categorical(y_train.astype("float32"), num_classes=10)
    y_val = y_train[-10000:]
    y_train = y_train[:-10000]
    y_train = tf.concat([y_train]*5, axis=0)

    y_test  = to_categorical(y_test.astype("float32"), num_classes=10)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def train():
    batch_size = 50

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()

    log_dir = "logs/mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,patience=10),
        #slow learning rate if model does not improve
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    model = keras.Sequential(layers=[

        keras.layers.Convolution2D(5, 6, activation="gelu", input_shape=(28,28,1)),
        #keras.layers.Convolution2D(4, (5,5), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    #Train model on training_data and report back using the validation data
    #Use the log likelihood cost function and the SGD optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=50,\
         batch_size=batch_size, verbose=1, callbacks=callbacks)

    


    #model.evaluate(test_data[0], test_data[1], batch_size=batch_size)
    model.summary()

    #save model
    #model.save("tf_mnist_50epochs_conv_pool_flatten_dense_adam.h5")

    #TODO: Add callback to slow learning rate, increase epoch size
    #TODO: Test images, save model as well


def vectorized_result(i): #Turns a number into a model output
    #Get a tensor of shape (10,1) of zeros tensorflow
    v = np.zeros((1,10))
    #Set the index of the number to 1
    v[0][i] = 1.0
    return v

def evaluate_image(path, actual, inverse, network): 
    
    with Image.open(path).convert('L') as img:

        # Resize the image to 28x28 pixels
        img = img.resize((28, 28))

        # Convert the image to a numpy array
        #arr = tf.array(img)
        #convert to tensor instead 
        arr = np.array(img)
        arr = arr.reshape((1, 28, 28, 1)).astype('float32') / 255.0

        # Invert the pixel values (if needed)
        if inverse:
            arr = 1.0 - arr

        print(network.predict(arr).shape)
        for i,element in enumerate(network.predict(arr)[0]):
            add_str = ""
            if i == actual:
                add_str = "*"
            print(f'{i}{add_str}\t{element:.6f}')
        network.evaluate(arr, vectorized_result(actual))
            
        print("\n")

def pretrain():
    #Load the model
    model = keras.models.load_model("Tensorflow_Models/tf_mnist_50epochs_conv_pool_flatten_dense.h5")
    
    #load mnist
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    
    model.evaluate(x_test, y_test)

    evaluate_image("CollinMNIST/2.png", 2, True, model)
    evaluate_image("CollinMNIST/0.webp", 0, False, model)
    evaluate_image("CollinMNIST/2 better.png", 2, True, model)
    evaluate_image("CollinMNIST/5.png", 5, True, model)
    evaluate_image("CollinMNIST/9.png", 9, True, model)
    evaluate_image("CollinMNIST/9 better.png", 9, True, model)
    evaluate_image("CollinMNIST/8.png", 8, True, model)
    

def main():
    train()
    #pretrain()

if __name__ == "__main__":
    main()