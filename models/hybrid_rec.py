import numpy as np
from .base_rec import BaseRecommender


class HybridRecommender(BaseRecommender):
    """
    3-way hybrid recommender: CF + Popularity + Content.

    final_score = α * CF_norm + β * Pop_norm + γ * Content_norm

    where normalization is done PER USER over that user's candidate set.
    """

    def __init__(
        self,
        cf_model,
        pop_model,
        content_model,
        alpha_cf: float = 0.6,
        alpha_pop: float = 0.25,
        alpha_content: float = 0.15,
    ):
        super().__init__(name="HybridCF+Pop+Content")
        self.cf_model = cf_model
        self.pop_model = pop_model
        self.content_model = content_model

        total = alpha_cf + alpha_pop + alpha_content
        if total <= 0:
            total = 1.0
        self.alpha_cf = alpha_cf / total
        self.alpha_pop = alpha_pop / total
        self.alpha_content = alpha_content / total

    def fit(self, X, y):
        """
        No-op: underlying models are trained separately.
        """
        return self

    def _norm(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        else:
            return np.zeros_like(x)

    def predict(self, X):
        """
        Global blend (no per-user normalization). Mainly for API completeness.
        For ranking, we actually rely on recommend().
        """
        cf_scores = self.cf_model.predict(X)
        pop_scores = self.pop_model.predict(X)
        cont_scores = self.content_model.predict(X)

        return (
            self.alpha_cf * cf_scores
            + self.alpha_pop * pop_scores
            + self.alpha_content * cont_scores
        )

    def recommend(self, X_candidates, candidate_ids, k=None):
        """
        Recommend items for a single user.

        X_candidates: all (playerid, gameid) rows for that one user.
        """
        cf_scores = self.cf_model.predict(X_candidates)
        pop_scores = self.pop_model.predict(X_candidates)
        cont_scores = self.content_model.predict(X_candidates)

        cf_norm = self._norm(cf_scores)
        pop_norm = self._norm(pop_scores)
        cont_norm = self._norm(cont_scores)

        final_scores = (
            self.alpha_cf * cf_norm
            + self.alpha_pop * pop_norm
            + self.alpha_content * cont_norm
        )

        order = np.argsort(-final_scores)
        if k is not None:
            order = order[:k]

        return [candidate_ids[i] for i in order]
