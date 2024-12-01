import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.coefficients : np.ndarray = np.ndarray([])
    
    def _add_bias_feature(self, X):
        # Add a column of ones as the bias feature (intercept)
        return np.insert(X, 0, 1, axis=1)
    
    def _normalize_features(self, X):
        # Normalize features by subtracting mean and dividing by standard deviation
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        return (X - mean_X) / std_X
    
    def fit(self, X, y):
        if self.copy_X:
            X = np.copy(X)
        
        if self.fit_intercept:
            X = self._add_bias_feature(X)
        
        if self.normalize:
            X = self._normalize_features(X)
        
        # Initialize coefficients
        self.coefficients = np.zeros(X.shape[1])
        
        # Gradient Descent to optimize coefficients
        alpha = 0.01  # Learning rate
        num_iterations = 1000  # Number of iterations
        m = len(y)  # Number of training examples
        
        for _ in range(num_iterations):
            predictions = np.dot(X, self.coefficients)
            error = predictions - y
            gradient = (1 / m) * np.dot(X.T, error)
            self.coefficients -= alpha * gradient
        
    def predict(self, X):
        if self.fit_intercept:
            X = self._add_bias_feature(X)
        
        if self.normalize:
            X = self._normalize_features(X)
        
        return np.dot(X, self.coefficients)
