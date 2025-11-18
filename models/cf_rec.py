import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .base_rec import BaseRecommender

class CFRecommender(BaseRecommender):
    DEFAULT_N_COMPONENTS = 3
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, n_components=None, random_state=None):
        super().__init__(name="CollaborativeFiltering")
        self.n_components = n_components or self.DEFAULT_N_COMPONENTS
        self.random_state = random_state or self.DEFAULT_RANDOM_STATE
        self.user_id_map = {}
        self.item_id_map = {}
        self.user_means = {}
        self.global_mean = 0.5
        self.user_factors = None
        self.item_factors = None
        self.singular_values = None

    def fit(self, X_train, y_train):
        player_ids = X_train["playerid"].values
        game_ids = X_train["gameid"].values
        scores = y_train.values.astype(float)

        unique_users = np.unique(player_ids)
        unique_games = np.unique(game_ids)

        self.user_id_map = {u: i for i, u in enumerate(unique_users)}
        self.item_id_map = {g: i for i, g in enumerate(unique_games)}

        n_users, n_items = len(unique_users), len(unique_games)

        if n_users < 2 or n_items < 2:
            self.global_mean = float(scores.mean())
            return

        user_idx = np.array([self.user_id_map[u] for u in player_ids])
        game_idx = np.array([self.item_id_map[g] for g in game_ids])

        R = csr_matrix((scores, (user_idx, game_idx)), shape=(n_users, n_items))
        self.global_mean = float(scores.mean())

        # compute user means
        for i in range(n_users):
            row = R.getrow(i).data
            self.user_means[i] = float(row.mean()) if row.size > 0 else self.global_mean

        # mean-center
        R_centered = R.copy().tolil()
        for i in range(n_users):
            for j in R.getrow(i).indices:
                R_centered[i, j] -= self.user_means[i]
        R_centered = R_centered.tocsr()

        k = min(self.n_components, min(n_users, n_items)-1)
        if k <= 0:
            return

        U, s, Vt = svds(R_centered, k=k, random_state=self.random_state)
        self.user_factors = U
        self.singular_values = s
        self.item_factors = Vt.T

    def predict(self, X):
        if self.user_factors is None:
            return np.full(len(X), self.global_mean)

        preds = np.full(len(X), self.global_mean)
        user_ids = X["playerid"].values
        game_ids = X["gameid"].values

        for i, (u, g) in enumerate(zip(user_ids, game_ids)):
            if u not in self.user_id_map or g not in self.item_id_map:
                continue

            ui = self.user_id_map[u]
            gi = self.item_id_map[g]
            uf = self.user_factors[ui] * self.singular_values
            pred = self.user_means[ui] + np.dot(uf, self.item_factors[gi])
            preds[i] = np.clip(pred, 0.0, 1.0)

        return preds
