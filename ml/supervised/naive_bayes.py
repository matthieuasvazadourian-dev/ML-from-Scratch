import numpy as np


# Bernoulli Naive Bayes for binary features (e.g. spam classification: word present or not)
# Assumes conditional independence between features given the class label (Naive Bayes assumption)

class NaiveBayes:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.phi_j_given_Y1 = None  # P(Xj=1 | Y=1), shape (n,)
        self.phi_j_given_Y0 = None  # P(Xj=1 | Y=0), shape (n,)
        self.phi_Y = None           # P(Y=1), scalar

    def fit(self, X, Y):
        Y = Y.reshape(-1)
        m, n = X.shape

        self.phi_Y = np.mean(Y == 1)

        # Laplace smoothing with k=2 (binary features)
        self.phi_j_given_Y1 = (np.sum(X[Y == 1], axis=0) + 1) / (np.sum(Y == 1) + 2)
        self.phi_j_given_Y0 = (np.sum(X[Y == 0], axis=0) + 1) / (np.sum(Y == 0) + 2)

        return self

    def predict_proba(self, X):
        # P(X|Y=k) = prod_j P(Xj|Y=k) via the Naive Bayes assumption
        p_X_Y1 = np.prod(self.phi_j_given_Y1 ** X * (1 - self.phi_j_given_Y1) ** (1 - X), axis=1)
        p_X_Y0 = np.prod(self.phi_j_given_Y0 ** X * (1 - self.phi_j_given_Y0) ** (1 - X), axis=1)

        # P(Y=1|X) via Bayes' rule
        return (self.phi_Y * p_X_Y1) / (self.phi_Y * p_X_Y1 + (1 - self.phi_Y) * p_X_Y0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def score(self, X, Y):
        return np.mean(self.predict(X) == Y.reshape(-1))
