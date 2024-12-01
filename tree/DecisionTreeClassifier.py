import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        # Stopping conditions
        if depth == self.max_depth or num_samples < self.min_samples_split or num_classes == 1:
            return self._create_leaf_node(y)
        
        # Find best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return self._create_leaf_node(y)
        
        feature_idx, threshold = best_split
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {'feature_index': feature_idx,
                'threshold': threshold,
                'left_subtree': left_subtree,
                'right_subtree': right_subtree}
    
    def _find_best_split(self, X, y):
        best_split = None
        best_score = -np.inf
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold
                
                if self.criterion == 'gini':
                    score = self._gini_impurity(y, left_indices, right_indices)
                elif self.criterion == 'entropy':
                    score = self._entropy(y, left_indices, right_indices)
                
                if score > best_score:
                    best_score = score
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def _gini_impurity(self, y, left_indices, right_indices):
        gini_left = self._calculate_gini(y[left_indices])
        gini_right = self._calculate_gini(y[right_indices])
        total_samples = len(y)
        
        return (len(y[left_indices]) / total_samples) * gini_left + \
               (len(y[right_indices]) / total_samples) * gini_right
    
    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini
    
    def _entropy(self, y, left_indices, right_indices):
        entropy_left = self._calculate_entropy(y[left_indices])
        entropy_right = self._calculate_entropy(y[right_indices])
        total_samples = len(y)
        
        return (len(y[left_indices]) / total_samples) * entropy_left + \
               (len(y[right_indices]) / total_samples) * entropy_right
    
    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding small epsilon to avoid log(0)
        return entropy
    
    # def _create_leaf_node(self, y):
    #     unique_classes, class_counts = np.unique(y, return_counts=True)
    #     majority_class = unique_classes[np.argmax(class_counts)]
    #     return {'class': majority_class}

    def _create_leaf_node(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        if len(class_counts) == 0:
            # If class_counts is empty, handle this case appropriately (e.g., return a default class)
            # Here, we can return a default class or raise an exception depending on the desired behavior
            raise ValueError("No class counts found in leaf node.")
        
        majority_class = unique_classes[np.argmax(class_counts)]
        return {'class': majority_class}

    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if 'class' in node:
            return node['class']
        
        feature_idx = node['feature_index']
        threshold = node['threshold']
        
        if x[feature_idx] <= threshold:
            return self._traverse_tree(x, node['left_subtree'])
        else:
            return self._traverse_tree(x, node['right_subtree'])
