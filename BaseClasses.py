from abc import ABC, abstractmethod
import numpy as np

class Estimator(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
    @abstractmethod
    def set_params(self, **params):
        pass

class Predictor(Estimator, ABC):
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def fit_predict(self, X, y) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def score(self, X, y) -> float:
        pass
