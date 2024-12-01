import numpy as np

class DecisionStumpRegressor:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.prediction = None
    
    def fit(self, X, y):
        self.y = y
        num_features = X.shape[1]
        best_mse = float('inf')
        
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                # Split the data based on the current feature and threshold
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold
                
                # Calculate mean squared error (MSE) for this split
                mse = np.mean((y[left_indices] - np.mean(y[left_indices]))**2) + np.mean((y[right_indices] - np.mean(y[right_indices]))**2)
                
                # Update the best split if we found a lower MSE
                if mse < best_mse:
                    best_mse = mse
                    self.feature_index = feature_idx
                    self.threshold = threshold
                    self.prediction = np.mean(y)
    
    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.ones(num_samples) * self.prediction
        
        for i in range(num_samples):
            if X[i, self.feature_index] <= self.threshold:
                predictions[i] = np.mean(self.y[X[:, self.feature_index] <= self.threshold])
            else:
                predictions[i] = np.mean(self.y[X[:, self.feature_index] > self.threshold])
        
        return predictions
