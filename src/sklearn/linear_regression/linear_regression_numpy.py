import numpy as np
# from "./asse"

"""
Linear Regression from scratch (Numpy + Gradient Descent)
"""

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000, verbose: bool = False):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        errors = y_true - y_pred
        return float(np.sum(errors ** 2) / 2)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: shape (m_samples, n_features)
        y: shape (m_samples)
        """

        X = np.ndarray(X, dtype=float)
        y = np.ndarray(y, dtype=float)
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            y_pred = self.w @ X + self.b


            cost = self._cost(y, y_pred)
            self.loss_history.append(cost)

            errors = y - y_pred
            dW = (X.T @ errors) / m
            db = np.sum(errors) / m

            if self.verbose and epoch % max(1, self.n_epochs // 10) == 0:
                 print(f"Epoch {epoch:5d} | Cost: {cost:.6f}")

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b
