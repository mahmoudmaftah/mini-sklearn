import sys
sys.path.append('..')
from _helpers.distance_metrics import euclidean_distance, manhattan_distance, minkowski_distance, cosine_similarity
import numpy as np

class KNNClassifier:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = np.ndarray([])
        self.y_train = np.ndarray([])

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_test_samples = X_test.shape[0]
        y_pred = np.zeros(num_test_samples, dtype=int)

        for i in range(num_test_samples):
            # Calculate distances between the current test sample and all training samples
            if self.distance_metric == 'euclidean':
                distances = np.array([euclidean_distance(X_test[i], x) for x in self.X_train])
            elif self.distance_metric == 'manhattan':
                distances = np.array([manhattan_distance(X_test[i], x) for x in self.X_train])
            elif self.distance_metric == 'minkowski':
                distances = np.array([minkowski_distance(X_test[i], x) for x in self.X_train])
            elif self.distance_metric == 'cosine':
                distances = np.array([cosine_similarity(X_test[i], x) for x in self.X_train])

            # Get indices of the k nearest neighbors
            nearest_neighbor_indices = np.argsort(distances)[:self.k]

            # Get labels of the k nearest neighbors
            k_nearest_labels = self.y_train[nearest_neighbor_indices]

            # Perform majority voting to determine the predicted class
            most_common_label = np.argmax(np.bincount(k_nearest_labels))
            y_pred[i] = most_common_label

        return y_pred
