import numpy as np
import random
import gzip
import pickle
from PIL import Image

def sigmoid(z):
    z = np.clip(z, -10, 10) #Make sure output doesnt explode
    z = np.where(z < 0, np.exp(z) / (1 + np.exp(z)), 1 / (1 + np.exp(-z))) 
    #further ensure output doesn't explode
    #z = np.divide(1        , 1 + np.exp(-1 * z), out=np.zeros_like(z), where=z>=0) + \
    #    np.divide(np.exp(z), 1 +      np.exp(z), out=np.zeros_like(z), where=z<0)

    return z

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z): #z is the full vector WA+B for a layer 
    z -= np.max(z) #Shift z so that it isn't super big and causes NaN
    exps = np.exp(z)
    return exps / np.sum(exps) #Get the full vector out

def softmax_prime(z):
    s = softmax(z).reshape(-1,1) #reshape into column vector
    jacobian = np.diagflat(s) - np.dot(s, s.T) 
    #np.dot gives matrix of S_i * S_j
    #np.diagflat makes it so that s is on the diagonal
    #To give S_i - S_i S_j or simply -S_i S_j off diagonal
    # = S_i (\delta_ij - S_j)
    #Which is the correct expression for the derivative of softmax
    return np.dot(jacobian, z) #Finally apply the Jacobian to z to get the output

def quadratic_cost(network_output, actual):
    return pow(network_output - actual, 2) / 2

def qc_prime(network_output, actual):
    return network_output - actual

def cross_entropy(network_output, actual):
    #network_output += 1e-9 #Add a small amount to prevent ln(0)
    return -1 * (actual * np.log(network_output) \
        + (1 - actual) * np.log(1 - network_output))

def ce_prime(network_output, actual):
    #network_output += 1e-9 #small amount to avoid divide by zero
    return (network_output - actual) / (network_output * (1 - network_output))

def log_cost(network_output,actual): #for softmax my beloved
    return log(network_output/actual)

def lc_prime(network_output, actual):
    return 1/network_output

def l2(weights, lmbda):
    return  lmda * pow(weights, 2) / 2 

def l2_prime(weights, lmbda):
    print(weights, lmbda)
    weights = np.array(weights)
    return weights * lmbda

class Layer: 
    def __init__(self, in_nodes, out_nodes, activation_function, activation_function_derivative):
        #self.nodes = node_count
        self.activation = activation_function
        self.activation_prime = activation_function_derivative
        self.biases = np.random.randn(out_nodes, 1) #Array of vectors
        self.weights = np.random.randn(out_nodes,in_nodes)/np.sqrt(in_nodes)
        print(self.weights.shape)

    def sigmoid_layer(in_nodes, out_nodes):
        return Layer(in_nodes, out_nodes, sigmoid, sigmoid_prime)

    def softmax_layer(in_nodes, out_nodes):
        return Layer(in_nodes, out_nodes, softmax, softmax_prime)

    def __call__(self, x):
        z = np.dot(self.weights,x) + self.biases
        a = self.activation(z)
        return z, a 

class Network:
    def __init__(self, layer_list, cost, cost_prime, lmda, clipping_threshold):
        self.layers = layer_list
        self.cost = cost
        self.cost_prime = cost_prime
        #self.biases = [np.random.randn(y.nodes, 1) for y in self.layers[1:]] #Array of vectors
        #self.weights = [np.random.randn(y.nodes,x.nodes)/np.sqrt(x.nodes) \
        #    for x,y in zip(self.layers[:-1], self.layers[1:])] 
            #Add sqrt to make the weights smaller and thus more predictive
        self.lmda = lmda
        self.clipping_threshold = clipping_threshold
        #Array of matrices
        #np.randn(y,x) produces random values from the normal distribution in a y by x matrix
        #layers[1:] and layers[:-1] produces the matrix sizes by getting the input size
        #from the previous layer (x doesn't include the last, first matched up with second)

    def feed_forward(self, input, store_intermediates = False):
        #Optionally stores intermediate layers for use in the backprop algorithm
        output = input

        if store_intermediates:
            zs = []
            activations = [input]

        for layer in self.layers:
            z, output = layer(output)
            
            if store_intermediates:
                zs.append(z)
                activations.append(output)

        if store_intermediates:
            return zs, activations
        else:
            return output
    
    def gradient_descent(self, training_data, epochs, minibatch_size, step, fail_n, test_data=None):
        n = len(training_data)

        if test_data:
            n_test = len(test_data)   

        epoch_score = []
        for epoch in range(epochs):
            random.shuffle(training_data)    
            minibatches = [training_data[x : x + minibatch_size] \
                for x in range(0, n, minibatch_size)]

            for batch in minibatches:
                self._update_minibatch(batch, step, n, fail_n)
            
            epoch_score.append(self.evaluate(test_data) / n_test)
            if len(epoch_score) > fail_n + 1 and \
                np.average(epoch_score[-fail_n:]) < epoch_score[-fail_n-1]:
                step /= 2
                #print(f'Epoch {epoch}: {self.evaluate(test_data)} / {n_test}')
                #break

            if test_data:
                print(f'Epoch {epoch}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {epoch} finished.')

            
    def _update_minibatch(self, batch, step, n, fail_n):
        nabla_b = [np.zeros(l.biases.shape) for l in self.layers]
        nabla_w = [np.zeros(l.weights.shape) for l in self.layers]

        for x, y in batch:
            db, dw = self._backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, db)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dw)]
        
        coeff = step / len(batch) #Learning rate + average out nablas

        for l, nw, nb in zip(self.layers, nabla_w, nabla_b):
            l.weights = (1 - step*self.lmda/n) * l.weights - coeff * nw
            l.biases  = l.biases - coeff * nb


    def _backprop(self, input, actual):
        nabla_b = [np.zeros(l.biases.shape) for l in self.layers]
        nabla_w = [np.zeros(l.weights.shape) for l in self.layers]

        zs, activations = self.feed_forward(input, True)

        #Choose delta such that it follows the direction of \nabla C
        #For the last layer
        #\partial{C}{b_n} = C'(f(z), actual) f_n'(z) \partial{z}{b_n} 
        #\partial{z}{b_n} = \partial{b_n} w * activations[-2] + b_n = 1
        #Where f(z) is just the model output and z is the model output before the last activation
        #\partial{C}{w_n} = C'(f(z), actual) f_n'(z) \partial{z}{w_n}
        #\partial{z}{w_n} = \partial{w_n} w * activations[-2] + b_n = activations[-2]
        #This continues on for each, with delta applying to the next step up until
        #The partial derivatives with respect to w_l and b_l
        #Which then gives delta = weights_l+1 * delta * f_l'(z_l) since (WA + B)' = W
        #And then you can do the same expressions for the nablas 
        delta = self.cost_prime(activations[-1], actual) * \
            self.layers[-1].activation_prime(zs[-1]) 
        #Sdelta = np.clip(delta, -self.clipping_threshold, self.clipping_threshold)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #Transpose matrix to fix the dimensions of delta as a layers[-1].nodes
        #Also so that we get a matrix out instead of a vector
        #Should be equivalent to np.outer(delta, activations[-2])

        for l in range(2, len(self.layers)):
            #Once again transpose because we are going backwards through the model
            delta = np.dot(self.layers[-l+1].weights.transpose(), delta) \
                * self.layers[-l].activation_prime(zs[-l])
            #delta = np.clip(delta, -self.clipping_threshold, self.clipping_threshold)

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data ]
        #print(test_results)
        return sum(int(x == y) for x, y in test_results)

    def save(self, model_name):  
        with open(f'{model_name}.pickle', 'wb') as file:
            pickle.dump(self, file)

    def load_from(model_name):
        with open(f'{model_name}.pickle', 'rb') as file:
            return pickle.load(file)

def vectorized_result(i): #Turns a number into a model output
    v = np.zeros((10,1))
    v[i] = 1.0
    return v

def mnist_loader():
    with gzip.open("MNIST/mnist.pkl.gz", 'rb') as file:
        tr_d, va_d, te_d = pickle.load(file, encoding="latin1")

    #Copied from neuralnetworksanddeeplearning.com
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data) #Data now usable!

def evaluate_image(path, actual, inverse, network): 
    
    with Image.open(path).convert('L') as img:

        # Resize the image to 28x28 pixels
        img = img.resize((28, 28))

        # Convert the image to a numpy array
        arr = np.array(img)

        # Normalize the pixel values to be between 0 and 1
        arr = arr.astype('float32') / 255.0

        # Flatten the array into a 1D array of length 784 (28*28)
        arr = arr.flatten()
        arr = np.reshape(arr, (784,1))
        #arr.transpose()
        #print (arr.shape)
        # Reshape the flattened array back into a 28x28 matrix
        #arr = arr.reshape((28, 28))

        # Invert the pixel values (if needed)
        if inverse:
            arr = 1.0 - arr

        for i,element in enumerate(network.feed_forward(arr)):
            add_str = ""
            if i == actual:
                add_str = "*"
            print(f'{i}{add_str}\t{element[0]:.6f}')
            
        print("\n")
        #print(network.feed_forward(arr))

def main():
    layers = [
        Layer.sigmoid_layer(784,30),
        Layer.sigmoid_layer(30,10),
        Layer.sigmoid_layer(10,10)
    ]

    training_data, validation_data, test_data = mnist_loader()
    network = Network(layers, cross_entropy, ce_prime, 4, 500)
    network.gradient_descent(training_data, 30, 50, 1, 10, test_data)
    #network.save("mnist_sigmoid_784_100_10_crossentropy_model")

    #network = Network.load_from("mnist_sigmoid_784_100_10_model")
    evaluate_image("CollinMNIST/2.png", 2, True, network)
    evaluate_image("CollinMNIST/0.webp", 0, False, network)
    evaluate_image("CollinMNIST/2 better.png", 2, True, network)
    evaluate_image("CollinMNIST/5.png", 5, True, network)
    evaluate_image("CollinMNIST/9.png", 9, True, network)
    evaluate_image("CollinMNIST/9 better.png", 9, True, network)
    evaluate_image("CollinMNIST/8.png", 8, True, network)


if __name__ == "__main__":
    main()