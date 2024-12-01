import numpy as np

class MLPRegressor:
    def __init__(self, input_size = 4, hidden_sizes = 1, output_size = 3, activation='relu'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1  # Including output layer
        self.activation = activation
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for hidden layers
        prev_size = input_size
        for size in hidden_sizes:
            self.weights.append(np.random.randn(size, prev_size))
            self.biases.append(np.zeros((size, 1)))
            prev_size = size
        
        # Initialize weights and biases for output layer
        self.weights.append(np.random.randn(output_size, prev_size))
        self.biases.append(np.zeros((output_size, 1)))
    
    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)  # ReLU activation
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))  # Sigmoid activation
        else:
            raise ValueError("Activation function not supported.")
    
    def _linear(self, x):
        return x  # Linear activation for output layer
    
    def _forward(self, x):
        a = x
        self.layer_inputs = [a]
        self.layer_outputs = []
        
        # Forward propagation through hidden layers
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self._activate(z)
            self.layer_inputs.append(z)
            self.layer_outputs.append(a)
        
        # Forward propagation through output layer (linear activation)
        z_out = np.dot(self.weights[-1], a) + self.biases[-1]
        a_out = self._linear(z_out)
        self.layer_inputs.append(z_out)
        self.layer_outputs.append(a_out)
        
        return a_out
    
    def _mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def _backward(self, x, y_true, y_pred):
        # Compute gradients using backpropagation
        m = x.shape[1]
        grad_weights = [np.zeros_like(W) for W in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # Compute gradient for output layer
        delta = -(y_true - y_pred)
        grad_weights[-1] = np.dot(delta, self.layer_outputs[-2].T) / m
        grad_biases[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # Backpropagate gradients through hidden layers
        for l in range(self.num_layers - 2, -1, -1):
            delta = np.dot(self.weights[l+1].T, delta) * (self.layer_outputs[l+1] > 0)
            if l == 0:
                grad_weights[l] = np.dot(delta, x.T) / m
            else:
                grad_weights[l] = np.dot(delta, self.layer_outputs[l-1].T) / m
            grad_biases[l] = np.sum(delta, axis=1, keepdims=True) / m
        
        return grad_weights, grad_biases
    
    def train(self, X_train, y_train, learning_rate=0.01, num_epochs=1000):
        for epoch in range(num_epochs):
            # Forward propagation
            y_pred = self._forward(X_train)
            
            # Compute loss
            loss = self._mean_squared_error(y_train, y_pred)
            
            # Backpropagation
            grad_weights, grad_biases = self._backward(X_train, y_train, y_pred)
            
            # Update weights and biases using gradient descent
            for i in range(self.num_layers):
                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i] -= learning_rate * grad_biases[i]
            
            # Print training progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict(self, X_test):
        # Make predictions using the trained model
        return self._forward(X_test)
