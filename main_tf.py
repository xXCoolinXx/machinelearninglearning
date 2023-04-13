import tensorflow as tf 
from tensorflow import keras

def main():
    model = keras.Sequential(
        keras.layers.Convolution2D(4, (24,24), activation="relu"),
        keras.layers.MaxPooling(pool_size(2, 2), strides=2),
        keras.layers.Dense(10, activation="softmax")
    )

    model.add(keras.Input(shape=(28,28)))
    model.summary()
    

if __name__ == "__main___":
    main()