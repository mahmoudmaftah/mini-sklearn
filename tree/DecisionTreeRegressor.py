import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y):
        if self.criterion == 'mse':
            self.tree = self._build_tree(X, y, depth=0, split_criterion=self._calculate_mse)
        elif self.criterion == 'mae':
            self.tree = self._build_tree(X, y, depth=0, split_criterion=self._calculate_mae)
        else:
            raise ValueError(f"Unknown criterion '{self.criterion}'. Please choose 'mse' or 'mae'.")
    
    def _build_tree(self, X, y, depth, split_criterion):
        num_samples, num_features = X.shape
        
        # Stopping conditions
        if depth == self.max_depth or num_samples < self.min_samples_split:
            return self._create_leaf_node(y)
        
        # Find best split using the specified criterion
        best_split = self._find_best_split(X, y, split_criterion)
        if best_split is None:
            return self._create_leaf_node(y)
        
        feature_idx, threshold = best_split
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1, split_criterion)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1, split_criterion)
        
        return {'feature_index': feature_idx,
                'threshold': threshold,
                'left_subtree': left_subtree,
                'right_subtree': right_subtree}
    
    def _find_best_split(self, X, y, split_criterion):
        best_split = None
        best_score = float('inf')  # Initialize best score with infinity for minimization
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold
                
                score = split_criterion(y[left_indices], y[right_indices])
                
                if score < best_score:
                    best_score = score
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def _calculate_mse(self, y_left, y_right):
        if len(y_left) == 0 or len(y_right) == 0:
            return float('inf')  # Return infinity if either side is empty
        
        mse_left = np.var(y_left)
        mse_right = np.var(y_right)
        total_samples = len(y_left) + len(y_right)
        
        return (len(y_left) / total_samples) * mse_left + (len(y_right) / total_samples) * mse_right
    
    def _calculate_mae(self, y_left, y_right):
        if len(y_left) == 0 or len(y_right) == 0:
            return float('inf')  # Return infinity if either side is empty
        
        mae_left = np.mean(np.abs(y_left - np.mean(y_left)))
        mae_right = np.mean(np.abs(y_right - np.mean(y_right)))
        total_samples = len(y_left) + len(y_right)
        
        return (len(y_left) / total_samples) * mae_left + (len(y_right) / total_samples) * mae_right
    
    def _create_leaf_node(self, y):
        return {'prediction': np.mean(y)}  # Store the mean of target values in the leaf node
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if 'prediction' in node:
            return node['prediction']
        
        feature_idx = node['feature_index']
        threshold = node['threshold']
        
        if x[feature_idx] <= threshold:
            return self._traverse_tree(x, node['left_subtree'])
        else:
            return self._traverse_tree(x, node['right_subtree'])
