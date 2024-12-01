import numpy as np

class DecisionStumpClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.prediction: int = 0
    
    def fit(self, X, y):
        num_features = X.shape[1]
        best_accuracy = 0
        
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                # Split the data based on the current feature and threshold
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold
                
                # Calculate accuracy for this split
                accuracy = np.mean(y[left_indices] == 0) if np.mean(y[left_indices] == 0) > 0.5 else np.mean(y[right_indices] == 1)
                
                # Update the best split if we found a higher accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.feature_index = feature_idx
                    self.threshold = threshold
                    
                    if np.mean(y[left_indices] == 0) > 0.5:
                        self.prediction = 0
                    else:
                        self.prediction = 1
    
    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)
        
        for i in range(num_samples):
            if X[i, self.feature_index] <= self.threshold:
                predictions[i] = self.prediction
            else:
                predictions[i] = 1 - self.prediction
        
        return predictions.astype(int)
