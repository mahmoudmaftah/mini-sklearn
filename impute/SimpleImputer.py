import numpy as np

class SimpleImputer:
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=None, copy=True,
                 add_indicator=False, keep_empty_features=False):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

        self.statistics_ = None
        self.indicator_ = None

    def fit(self, X):
        # Compute the statistics based on the strategy
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            from scipy.stats import mode
            self.statistics_ = mode(X, axis=0)[0].squeeze()
        elif self.strategy == 'constant':
            if self.fill_value is None:
                raise ValueError("fill_value must be specified for strategy='constant'.")
            self.statistics_ = self.fill_value

        # Return the imputer object
        return self

    def transform(self, X):
        # Check if copy is needed
        if self.copy:
            X = X.copy()

        # Replace missing values based on strategy
        if self.strategy == 'constant':
            X[np.isnan(X)] = self.statistics_
        else:
            missing_mask = np.isnan(X)
            X[missing_mask] = self.statistics_[missing_mask]

        # Generate indicator if add_indicator is True
        if self.add_indicator:
            self.indicator_ = np.isnan(X).astype(int)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {
            'missing_values': self.missing_values,
            'strategy': self.strategy,
            'fill_value': self.fill_value,
            'copy': self.copy,
            'add_indicator': self.add_indicator,
            'keep_empty_features': self.keep_empty_features
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
