"""
Random Forest model implementation for Prophet.
"""

from sklearn.ensemble import RandomForestRegressor

class RandomForestPredictor(RandomForestRegressor):
    """
    Random Forest predictor for Prophet.

    This is a wrapper around sklearn's RandomForestRegressor to maintain
    consistency with the Prophet API and add additional functionality.
    """

    def __init__(self, verbose=2, **kwargs):
        defaults = {
            "max_depth": 12,
            "n_jobs": -1,
        }

        # Update defaults with provided kwargs
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        super().__init__(verbose=verbose, **kwargs)

    def fit(self, X, y):
        """Fit the Random Forest model."""
        return super().fit(X, y)

    def predict(self, X):
        """Make predictions."""
        return super().predict(X)
