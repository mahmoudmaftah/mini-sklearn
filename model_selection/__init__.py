import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as shuffle_arrays
from sklearn.utils import indexable
from _helpers import _num_samples
from .StratifiedShuffleSplit import StratifiedShuffleSplit

def train_test_split(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Input samples.

    y : array-like, shape (n_samples,)
        The target variable.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
        If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance, or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns:
    X_train : array-like
        The training input samples.

    X_test : array-like
        The testing input samples.

    y_train : array-like
        The training target values.

    y_test : array-like
        The testing target values.
    """
    random_state = check_random_state(random_state)
    n_samples = _num_samples(X)

    if stratify is not None:
        cv = StratifiedShuffleSplit(test_size=test_size, train_size=train_size, random_state=random_state)
        train_idx, test_idx = next(cv.split(X, stratify))
    else:
        if shuffle:
            X, y = shuffle_arrays(X, y, random_state=random_state)

        if test_size is None and train_size is None:
            test_size = 0.25

        if test_size is None:
            if isinstance(train_size, float):
                test_size = 1.0 - train_size
            else:
                test_size = 0.25

        if isinstance(test_size, float):
            test_size = round(test_size * n_samples)

        if isinstance(train_size, float):
            train_size = round(train_size * n_samples)

        test_size = int(test_size)
        train_size = n_samples - test_size

        train_idx = slice(None, train_size)
        test_idx = slice(train_size, None)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test
