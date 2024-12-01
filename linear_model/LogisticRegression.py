import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, penalty='l2', C=1.0, max_iter=100, random_state=None, verbose=0,):
                #  class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                #  warm_start=False, multi_class='ovr', n_jobs=None, solver='liblinear'):
        self.penalty = penalty
        self.C = C
        # self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        # self.class_weight = class_weight
        # self.dual = dual
        # self.fit_intercept = fit_intercept
        # self.intercept_scaling = intercept_scaling
        # self.warm_start = warm_start
        # self.multi_class = multi_class
        self.verbose = verbose
        # self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        
        self.w = np.ndarray([])  # weights
        self.b = 0  # intercept

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        probabilities = self._sigmoid(z)
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        
        if self.penalty == 'l1':
            l1_regularization = np.sum(np.abs(self.w))
            loss += (1 / m) * (self.C * l1_regularization)
        elif self.penalty == 'l2':
            l2_regularization = np.sum(self.w ** 2)
            loss += (1 / (2 * m)) * (self.C * l2_regularization)
        
        return loss

    def _compute_gradients(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        probabilities = self._sigmoid(z)
        
        dw = (1 / m) * np.dot(X.T, (probabilities - y))
        db = (1 / m) * np.sum(probabilities - y)
        
        if self.penalty == 'l1':
            dw += (self.C / m) * np.sign(self.w)
        elif self.penalty == 'l2':
            dw += (self.C / m) * self.w
        
        return dw, db

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.w = np.random.randn(X.shape[1])
        self.b = 0
        
        for _ in range(self.max_iter):
            loss = self._compute_loss(X, y)
            dw, db = self._compute_gradients(X, y)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if self.verbose > 0:
                print(f"Iteration {_ + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        probabilities = self._sigmoid(z)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
