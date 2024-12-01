import numpy as np

class NaiveBayesClassifier:
    def __init__(self, use_laplace_smoothing=False, alpha=1.0):
        self.use_laplace_smoothing = use_laplace_smoothing
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = np.ndarray([])
        self.class_conditional_probs = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # Calculate class priors
        self.class_priors = np.zeros(num_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == c) / float(num_samples)

        # Calculate conditional probabilities for each feature
        self.class_conditional_probs = []
        for c in self.classes:
            X_c = X[y == c]
            class_probs = []
            for feature_idx in range(num_features):
                feature_values = np.unique(X[:, feature_idx])
                num_values = len(feature_values)
                feature_prob = np.zeros(num_values)
                for v in range(num_values):
                    count = np.sum(X_c[:, feature_idx] == feature_values[v])
                    if self.use_laplace_smoothing:
                        count += self.alpha
                        total_count = len(X_c) + num_values * self.alpha
                    else:
                        total_count = len(X_c)
                    feature_prob[v] = count / total_count
                class_probs.append(feature_prob)
            self.class_conditional_probs.append(class_probs)

    def predict(self, X):
        num_samples = X.shape[0]
        y_pred = np.zeros(num_samples, dtype=int)
        for i in range(num_samples):
            posteriors = []
            for c, class_prob in enumerate(self.classes):
                posterior = np.log(self.class_priors[c])
                for feature_idx, x in enumerate(X[i]):
                    if x in self.class_conditional_probs[c][feature_idx]:
                        prob = self.class_conditional_probs[c][feature_idx][np.where(
                            self.class_conditional_probs[c][feature_idx] == x)[0][0]]
                    else:
                        if self.use_laplace_smoothing:
                            prob = self.alpha / (len(self.class_conditional_probs[c][feature_idx]) + self.alpha)
                        else:
                            prob = 0  # Consider zero probability if not using Laplace smoothing
                    posterior += np.log(prob)
                posteriors.append(posterior)
            y_pred[i] = self.classes[np.argmax(posteriors)]
        return y_pred
