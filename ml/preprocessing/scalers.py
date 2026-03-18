import numpy as np


class StandardScaler:
    """Z-score normalisation: transforms features to zero mean and unit variance.

    Attributes:
        mu (np.ndarray): Per-feature mean computed during fit, shape (n,).
        std (np.ndarray): Per-feature standard deviation computed during fit, shape (n,).

    Example:
        >>> scaler = StandardScaler().fit(X_train)
        >>> X_train_scaled = scaler.transform(X_train)
        >>> X_test_scaled  = scaler.transform(X_test)
    """

    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, X):
        """Compute mean and standard deviation from training data.

        Args:
            X (np.ndarray): Training data, shape (m, n).

        Returns:
            self: Fitted scaler (enables method chaining).
        """
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std == 0] = 1
        return self

    def transform(self, X):
        """Apply z-score normalisation.

        Args:
            X (np.ndarray): Data to transform, shape (m, n).

        Returns:
            np.ndarray: Scaled data, shape (m, n).
        """
        return (X - self.mu) / self.std

    def fit_transform(self, X):
        """Fit and transform in one step.

        Args:
            X (np.ndarray): Training data, shape (m, n).

        Returns:
            np.ndarray: Scaled data, shape (m, n).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling back to the original space.

        Args:
            X_scaled (np.ndarray): Scaled data, shape (m, n).

        Returns:
            np.ndarray: Data in the original scale, shape (m, n).
        """
        return X_scaled * self.std + self.mu


class MinMaxScaler:
    """Min-max normalisation: scales features to the [0, 1] range.

    Attributes:
        min_ (np.ndarray): Per-feature minimum computed during fit, shape (n,).
        max_ (np.ndarray): Per-feature maximum computed during fit, shape (n,).
        range_ (np.ndarray): Per-feature range (max - min), shape (n,).

    Example:
        >>> scaler = MinMaxScaler().fit(X_train)
        >>> X_train_scaled = scaler.transform(X_train)
        >>> X_test_scaled  = scaler.transform(X_test)
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        """Compute per-feature min and max from training data.

        Args:
            X (np.ndarray): Training data, shape (m, n).

        Returns:
            self: Fitted scaler (enables method chaining).
        """
        self.max_ = np.max(X, axis=0)
        self.min_ = np.min(X, axis=0)
        self.range_ = self.max_ - self.min_
        # Avoid division by zero for constant features
        self.range_[self.range_ == 0] = 1
        return self

    def transform(self, X):
        """Apply min-max normalisation.

        Args:
            X (np.ndarray): Data to transform, shape (m, n).

        Returns:
            np.ndarray: Scaled data in [0, 1], shape (m, n).
        """
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        """Fit and transform in one step.

        Args:
            X (np.ndarray): Training data, shape (m, n).

        Returns:
            np.ndarray: Scaled data, shape (m, n).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling back to the original space.

        Args:
            X_scaled (np.ndarray): Scaled data in [0, 1], shape (m, n).

        Returns:
            np.ndarray: Data in the original scale, shape (m, n).
        """
        return X_scaled * self.range_ + self.min_
