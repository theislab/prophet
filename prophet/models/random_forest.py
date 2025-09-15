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

    def __init__(self, verbose=2, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def fit(self, X, y):
        """Fit the Random Forest model."""
        print(
            f"🌲 Training RandomForest with {X.shape[0]} samples and {X.shape[1]} features"
        )
        return super().fit(X, y)

    def predict(self, X):
        """Make predictions."""
        print(f"🔮 Making predictions on {X.shape[0]} samples")
        return super().predict(X)
