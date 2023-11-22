# Artificial-Neural-Network

# Algorithm 
1. Getting the weighted sum of inputs of a particular unit using the h(x) function we defined
earlier.
2. Plugging the value we get from step 1 into the activation function we have (f(a)=a in this
example) and using the activation value we get (i.e. the output of the activation function) as
the input feature for the connected nodes in the next layer.
3. If feeding forward happened using the following functions:
f(a) = a
4. Then feeding backward will happen through the partial derivatives of those functions. There
is no need to go through the working of arriving at these derivatives. All we need to know
is that the above functions will follow:
f'(a) = 1
J'(w) = Z . delta
5. Updating the weights.

# Code
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.biases_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.biases_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # Calculate hidden layer values
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.biases_hidden)
        
        # Calculate output layer values
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.biases_output)

    def backward_propagation(self, inputs, targets, learning_rate):
        # Calculate output layer errors
        output_errors = targets - self.output_layer
        output_delta = output_errors * self.sigmoid_derivative(self.output_layer)

        # Update output layer weights and biases
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.biases_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        # Calculate hidden layer errors
        hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(self.hidden_layer)

        # Update input-to-hidden layer weights and biases
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        self.biases_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for input_data, target_data in zip(inputs, targets):
                input_data = np.array([input_data])
                target_data = np.array([target_data])

                self.forward_propagation(input_data)
                self.backward_propagation(input_data, target_data, learning_rate)

    def predict(self, inputs):
        inputs = np.array([inputs])
        self.forward_propagation(inputs)
        return self.output_layer

#Example usage:
#Sample dataset for XOR problem
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

#Create and train the neural network
input_size = 2
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(inputs, targets, epochs=10000, learning_rate=0.1)

#Test the neural network
for test_input in inputs:
    prediction = nn.predict(test_input)
    print(f"For input {test_input}, prediction: {prediction}")

# Link to run if this copy paste is not working
https://replit.com/@vigneshm2021csb/Artificial-Neural-Network

# input
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# output
For input [0 0], prediction: [[0.03639068]]
For input [0 1], prediction: [[0.95929856]]
For input [1 0], prediction: [[0.95766694]]
For input [1 1], prediction: [[0.05649402]]


