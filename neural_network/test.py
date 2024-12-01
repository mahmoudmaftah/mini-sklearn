import numpy as np


class Node:
    def __init__(self, weight=1.0):
        self.weight = weight
        self.input_val = None

    def set_input(self, input_val):
        self.input_val = input_val

#-----------------------------------------------------------

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        
    def get_weights(self):
        return self.weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def backward(self, input_data, d_output_data):
        d_output_data = d_output_data * self.sigmoid_derivative(np.dot(self.weights, input_data) + self.biases)
        d_weights = np.dot(d_output_data, input_data.T)
        d_biases = np.sum(d_output_data, axis=1, keepdims=True)
        d_input_data = np.dot(self.weights.T, d_output_data)
        return d_input_data, d_weights, d_biases

    def forward(self, input_data):
        output_data = np.dot(self.weights, input_data) + self.biases
        return self.sigmoid(output_data)
    
    def categorical_crossentropy_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
        return loss

    def update_params(self, d_weights, d_biases, learning_rate):
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

#-----------------------------------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.output_size = output_size
        self.layer1 = Layer(input_size, hidden_size1)
        self.layer2 = Layer(hidden_size1, output_size)

    def forward(self, input_data):
        hidden_output1 = self.layer1.forward(input_data)
        output_data = self.layer2.forward(hidden_output1)
        return output_data

    def train(self, X, y, epochs, learning_rate):
        loss_val = []
        for epoch in range(epochs):
            total_loss = 0
            for i in range(X.shape[1]):
                input_data = X[:, i].reshape(-1, 1)
                y_true = y[:, i].reshape(-1, 1)
                hidden_output1 = self.layer1.forward(input_data)
                output_data = self.layer2.forward(hidden_output1)
                loss = self.layer2.categorical_crossentropy_loss(y_true, output_data)
                total_loss += loss
                loss_val.append(total_loss / X.shape[1])
                d_output_data = output_data - y_true
                d_hidden_output1, d_weights2, d_biases2 = self.layer2.backward(hidden_output1, d_output_data)
                _, d_weights1, d_biases1 = self.layer1.backward(input_data, d_hidden_output1)
                self.layer2.update_params(d_weights2, d_biases2, learning_rate)
                self.layer1.update_params(d_weights1, d_biases1, learning_rate)
            average_loss = total_loss / X.shape[1]
            print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}")
        return loss_val
    
    def predict(self, input_data_point):
        output_data_point = self.forward(input_data_point.reshape(-1, 1))
        return output_data_point[1, 0]
    
    def test(self, X_test, y_test):
        test_predictions = self.forward(X_test.T)
        test_predictions_classes = np.argmax(test_predictions, axis=0)
        y_test_classes = np.argmax(y_test.T, axis=0)
        accuracy = np.mean(test_predictions_classes == y_test_classes)
        return accuracy