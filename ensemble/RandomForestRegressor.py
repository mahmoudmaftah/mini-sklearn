from DecisionTreeRegressor import DecisionTreeRegressor
import numpy as np

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_features=None, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        n_samples, n_features = X.shape
        
        # Determine max_features if not specified
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))  # Use sqrt(n_features) as a rule of thumb
        
        for _ in range(self.n_estimators):
            # Randomly sample with replacement for bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Select random features
            random_features = np.random.choice(n_features, size=self.max_features, replace=False)
            X_bootstrap = X_bootstrap[:, random_features]
            
            # Train a decision tree with the bootstrap sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Append the trained tree to the forest
            self.forest.append(tree)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for tree in self.forest:
            # Make predictions for each tree
            tree_predictions = tree.predict(X)
            
            # Accumulate predictions from all trees
            predictions += tree_predictions
        
        # Compute the average prediction from all trees
        predictions /= len(self.forest)
        
        return predictions


