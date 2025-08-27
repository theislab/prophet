"""
Random Forest model implementation for Prophet.
"""

from sklearn.ensemble import RandomForestRegressor


class RandomForestPredictor(RandomForestRegressor):
    """
    Random Forest predictor for Prophet.

    This is a wrapper around sklearn's RandomForestRegressor to maintain
    consistency with the Prophet API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        """Fit the Random Forest model."""
        return super().fit(X, y)

    def predict(self, X):
        """Make predictions."""
        return super().predict(X)
