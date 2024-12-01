from statistics import median

class VotingRegressor:
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X_train, y_train):
        for name, regressor in self.estimators:
            regressor.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = []
        for name, regressor in self.estimators:
            predictions.append(regressor.predict(X_test))
        
        if self.voting == 'hard':
            final_predictions = median(predictions)
        elif self.voting == 'soft':
            if self.weights is None:
                self.weights = [1] * len(self.estimators)  # Use equal weights if not specified
            weighted_predictions = [w * pred for w, pred in zip(self.weights, predictions)]
            final_predictions = sum(weighted_predictions) / sum(self.weights)

        return final_predictions