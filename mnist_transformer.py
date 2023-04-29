import tensorflow as tf
import numpy as np
import keras
from keras import layers
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Hyperparameters
num_classes = 10
num_epochs = 100
batch_size = 50
num_heads = 8
projection_dim = 64 #Projection of the patches to a flat vector dimension
ff_layer_size = [
    projection_dim * 2, #Idk why it's 2 times but that's what's in the tutorial
    projection_dim,
    projection_dim,
]
num_transformer_layers = 2
input_size = (28, 28, 1)
image_size = (28, 28)
patch_size = 4
num_patches = (image_size[0] // patch_size) ** 2
stop_patience = 10
stop_delta = 0.0001
lr_reduction_patience = 5
lr_reduction_factor = 0.5

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
def augmentor():
    return keras.Sequential(layers=[
            layers.Normalization(),
            layers.Resizing(*image_size), #Star to unpack the tuple
            layers.RandomRotation(0.1), 
            layers.RandomZoom(0.1),
    ],
    )

def feed_forward(x, units, dropout=0.1):
    for unit in units:
        x = layers.Dense(units=unit, activation="gelu")(x)
        x = layers.Dropout(rate=dropout)(x)
    return x

def add_plus_norm(x, y):
    add = layers.Add()([x, y])
    return layers.LayerNormalization(epsilon=1e-6)(add)

def encoder(x, projection_dim):
    """
    Returns a transformer encoder for use in image processing
                    Add---------------+
                     |                |
               +-----|-----+          |
               |   Feed    |          |
               |  Forward  |          |
               +-----|-----+          |
                     |----------------+
    +------------Add + Norm            
    |                |                 
    |          +-----|-----+           
    |          | Multihead |           
    |          | Attention |           
    |          +-----|-----+           
    |                |                 
    +------------Normalize
                     |
               Encoded input  
    """
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)

    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1) 

    x2 = add_plus_norm(x, x1)

    ff = feed_forward(x, ff_layer_size)

    x3 = layers.Add()([x2, ff])

    return x3

def get_model():
    data_augmentation = augmentor()

    input = layers.Input(input_size)
    ainput = data_augmentation(input)
    patches = Patches(patch_size=patch_size)(ainput)

    #Turn variable into flat vector
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(patches)

    #Layer several transformers on top of each other for better results hopefully
    for i in range(num_transformer_layers):
        encoded_patches = encoder(encoded_patches, projection_dim)

    encoder_output = layers.Flatten()(encoded_patches) #Flatten down to a 1-Tensor to feed to the dense layer

    logits = layers.Dense(num_classes, activation="softmax")(encoder_output) #Softmax for classification

    model = keras.Model(inputs=input, outputs=logits)
    
    return model

def train():
    #x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    y_train = to_categorical(y_train.astype("float32"), num_classes=num_classes)
    y_test  = to_categorical(y_test .astype("float32"), num_classes=num_classes)

    model = get_model()

    #log_dir = "logs/mnist_transformer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = "/tmp/mnist_transformer/checkpoint"
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=stop_delta,patience=stop_patience),
        #slow learning rate if model does not improve
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
            factor=lr_reduction_factor, 
            patience=lr_reduction_patience,
            min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        )
        #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        x_train, y_train, 
        callbacks=callbacks,
        epochs=num_epochs, 
        batch_size=batch_size, 
        validation_data=(x_test, y_test)
    )

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

def load():
    checkpoint_filepath = "./tmp/mnist_transformer/checkpoint"
    
    model = get_model()

    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint.restore(checkpoint_filepath)
    #model.load_weights(checkpoint_filepath, by_name=True, skip_mismatch=True).expect_partial()


    model.summary()

    evaluate_image("CollinMNIST/2.png", 2, True, model)
    evaluate_image("CollinMNIST/0.webp", 0, False, model)
    evaluate_image("CollinMNIST/2 better.png", 2, True, model)
    evaluate_image("CollinMNIST/5.png", 5, True, model)
    evaluate_image("CollinMNIST/9.png", 9, True, model)
    evaluate_image("CollinMNIST/9 better.png", 9, True, model)
    evaluate_image("CollinMNIST/8.png", 8, True, model)

def main():
    load()

if __name__ == "__main__":
    main()
