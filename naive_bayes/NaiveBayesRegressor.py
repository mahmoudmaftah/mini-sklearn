import numpy as np

class NaiveBayesRegressor:
    def __init__(self, use_laplace_smoothing=False, alpha=1.0):
        self.use_laplace_smoothing = use_laplace_smoothing
        self.alpha = alpha  # Laplace smoothing parameter
        self.mean_y = None
        self.std_y = None
        self.feature_means = None
        self.feature_variances = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Calculate global mean and standard deviation of target variable y
        self.mean_y = np.mean(y)
        self.std_y = np.std(y)

        # Calculate mean and variance of each feature given the target variable y
        self.feature_means = np.zeros(num_features)
        self.feature_variances = np.zeros(num_features)

        for feature_idx in range(num_features):
            feature_values = np.unique(X[:, feature_idx])
            num_values = len(feature_values)

            # Laplace smoothing for feature values if enabled
            if self.use_laplace_smoothing:
                counts = np.array([np.sum(X[:, feature_idx] == value) for value in feature_values]) + self.alpha
                total_counts = num_samples + num_values * self.alpha
            else:
                counts = np.array([np.sum(X[:, feature_idx] == value) for value in feature_values])
                total_counts = num_samples

            # Calculate conditional mean and variance of the feature
            conditional_mean = np.sum(feature_values * counts) / total_counts
            conditional_variance = np.sum(((feature_values - conditional_mean) ** 2) * counts) / total_counts

            self.feature_means[feature_idx] = conditional_mean
            self.feature_variances[feature_idx] = conditional_variance

    def predict(self, X):
        num_samples, num_features = X.shape
        y_pred = np.zeros(num_samples)

        for i in range(num_samples):
            predicted_value = self.mean_y  # Start with global mean of y

            for feature_idx in range(num_features):
                x = X[i, feature_idx]
                conditional_mean = self.feature_means[feature_idx]
                conditional_variance = self.feature_variances[feature_idx]

                # Calculate likelihood of feature value given the conditional mean and variance
                likelihood = (1.0 / np.sqrt(2 * np.pi * conditional_variance)) * np.exp(
                    -((x - conditional_mean) ** 2) / (2 * conditional_variance))

                # Update predicted value based on the conditional likelihood
                if likelihood > 0:
                    predicted_value *= likelihood

            y_pred[i] = predicted_value

        return y_pred
