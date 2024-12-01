import numpy as np

class Normalizer:
    def __init__(self, norm='l2'):
        self.norm = norm
        
    def fit(self, X):
        # No actual fitting is needed for normalizer
        return self
    
    def transform(self, X):
        # Choose norm type
        if self.norm not in ('l1', 'l2', 'max'):
            raise ValueError("Invalid norm type. Choose 'l1', 'l2', or 'max'.")
        
        # Apply normalization based on selected norm
        if self.norm == 'l1':
            X_normalized = X / np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'l2':
            X_normalized = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))
        elif self.norm == 'max':
            X_normalized = X / np.max(np.abs(X), axis=1, keepdims=True)
        
        return X_normalized
    
    def fit_transform(self, X):
        # Fit to the data and transform it in one step
        return self.transform(X)
