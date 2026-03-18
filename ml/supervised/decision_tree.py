import numpy as np


class DecisionTree:
    def __init__(self, min_information_gain=0):
        self.min_information_gain = min_information_gain
        self.tree_root = None

    def compute_entropy(self, node, y):
        l = node.shape[0]
        if l == 0:
            return 0
        p1 = np.sum(y[node]) / l
        if p1 == 0 or p1 == 1:
            return 0
        return -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))

    def split(self, X, node, feature):
        mask = X[node, feature] == 1
        return node[mask], node[~mask]

    def compute_information_gain(self, parent_node, left_node, right_node, y):
        N, Nl, Nr = parent_node.shape[0], left_node.shape[0], right_node.shape[0]
        return self.compute_entropy(parent_node, y) - (
            Nl / N * self.compute_entropy(left_node, y)
            + Nr / N * self.compute_entropy(right_node, y)
        )

    def perform_best_split(self, X, y, parent_node):
        num_features = X.shape[1]
        inf_gain = np.array([
            self.compute_information_gain(parent_node, *self.split(X, parent_node, f), y)
            for f in range(num_features)
        ])
        best = np.argmax(inf_gain)
        return self.split(X, parent_node, best), best, inf_gain[best]

    def majority_class(self, node, y):
        return 1 if np.mean(y[node]) >= 0.5 else 0

    def build(self, X, y, node):
        (left_node, right_node), split_feature, inf_gain = self.perform_best_split(X, y, node)

        if inf_gain < self.min_information_gain or left_node.shape[0] == 0 or right_node.shape[0] == 0:
            return {"is_leaf": True, "prediction": self.majority_class(node, y)}

        return {
            "is_leaf": False,
            "split_feature": split_feature,
            "prediction": self.majority_class(node, y),
            "left_node": self.build(X, y, left_node),
            "right_node": self.build(X, y, right_node),
        }

    def fit(self, X, y):
        self.tree_root = self.build(X, y, np.arange(y.shape[0]))
        return self

    def _predict_one(self, x):
        node = self.tree_root
        while not node["is_leaf"]:
            node = node["left_node"] if x[node["split_feature"]] == 1 else node["right_node"]
        return node["prediction"]

    def predict(self, X):
        return np.array([self._predict_one(X[i]) for i in range(X.shape[0])])

    def score(self, X, y):
        return np.mean(self.predict(X) == y.reshape(-1))
