import numpy as np


class KMeans:
    def __init__(self, num_clusters, num_iterations=300):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.centroids = None  # shape (num_clusters, n)
        self.labels_ = None    # shape (m,) — cluster assignment per sample

    def _assign(self, X):
        # Vectorised distance: (m, 1, n) - (1, k, n) -> (m, k, n) -> (m, k)
        diffs = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        k, n = self.num_clusters, X.shape[1]
        new_centroids = np.empty((k, n))
        for i in range(k):
            assigned = X[labels == i]
            new_centroids[i] = np.mean(assigned, axis=0) if len(assigned) > 0 else X[np.random.randint(X.shape[0])]
        return new_centroids

    def fit(self, X):
        m = X.shape[0]
        # Initialise centroids by sampling k random points
        self.centroids = X[np.random.choice(m, size=self.num_clusters, replace=False)]

        for _ in range(self.num_iterations):
            self.labels_ = self._assign(X)
            self.centroids = self._update_centroids(X, self.labels_)

        self.labels_ = self._assign(X)
        return self

    def predict(self, X):
        return self._assign(X)

    def score(self, X):
        # Negative inertia: sum of squared distances to assigned centroids
        labels = self.predict(X)
        inertia = sum(np.sum((X[labels == k] - self.centroids[k]) ** 2) for k in range(self.num_clusters))
        return -inertia
