import numpy as np

class Imputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X):
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = np.nanmax(np.unique(X, return_counts=True), axis=1)[0]
        else:
            raise ValueError("Unknown imputation strategy: {}".format(self.strategy))

    def transform(self, X):
        if self.statistics_ is None:
            raise ValueError("Imputer has not been fitted.")
        if self.strategy == 'constant':
            return np.where(np.isnan(X), self.fill_value, X)
        else:
            return np.where(np.isnan(X), self.statistics_, X)

# Example usage:
data = np.array([[1, 2, np.nan],
                 [4, np.nan, 6],
                 [7, 8, 9]])

imputer = Imputer(strategy='mean')
imputer.fit(data)
imputed_data = imputer.transform(data)

print("Original data:")
print(data)
print("\nImputed data:")
print(imputed_data)
