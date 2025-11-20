import numpy as np
from .base_rec import BaseRecommender

class PopularityRecommender(BaseRecommender):
    """
    Simple popularity-based recommender.
    Scores games by how many unique players interacted with them
    in the last month of the training data.
    """
    def __init__(self, popularity_scores=None):
        super().__init__(name="Popularity")
        self.popularity_scores = popularity_scores or {}

    def fit(self, X_train, y_train):
        # Popularity model doesn't train on X/y.
        # Scores were precomputed in data_prep.
        pass

    def predict(self, X):
        game_ids = X["gameid"].values
        return np.array([
            self.popularity_scores.get(gid, 0.0)
            for gid in game_ids
        ], dtype=float)
