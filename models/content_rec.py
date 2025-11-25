import numpy as np
from .base_rec import BaseRecommender


class ContentRecommender(BaseRecommender):
    """
    Simple content-based recommender using primary_genre and per-player
    genre preferences.

    Idea:
      - For each player, compute how much they 'like' each genre
        based on TRAIN interactions (using `liked`).
      - For each game, we know its `primary_genre`.
      - Score(player, game) = blend of:
          * player preference for that game's genre
          * global average like rate for that game (or genre)
    """

    def __init__(self, data_prep, w_player=0.7, w_item=0.3):
        super().__init__(name="Content")
        self.data_prep = data_prep
        self.w_player = w_player
        self.w_item = w_item

        self.user_genre_pref = {}   # {playerid: {genre: pref}}
        self.game_base_score = {}   # {gameid: base_like_rate}
        self.game_genre = {}        # {gameid: primary_genre}
        self.global_genre_like = {} # {genre: like_rate}
        self.global_mean = 0.0

    def fit(self, X_train, y_train):
        """
        We ignore X_train/y_train and build from data_prep.train_df.
        """
        train = self.data_prep.train_df.copy()

        # Only rows with a valid game and genre
        train = train.dropna(subset=["gameid", "primary_genre"])

        # Global mean like rate (for fallback)
        self.global_mean = float(train["liked"].mean()) if len(train) > 0 else 0.0

        # Per-game base like rate (how well the game does overall)
        game_stats = train.groupby("gameid")["liked"].mean()
        self.game_base_score = game_stats.to_dict()

        # Game → primary_genre mapping
        game_genre = train.groupby("gameid")["primary_genre"].first()
        self.game_genre = game_genre.to_dict()

        # Per-genre global like rate
        genre_like = train.groupby("primary_genre")["liked"].mean()
        self.global_genre_like = genre_like.to_dict()

        # Per-player, per-genre like rate
        # e.g. how often player liked games of that genre in train
        user_genre = (
            train.groupby(["playerid", "primary_genre"])["liked"]
            .mean()
            .reset_index()
        )

        # Build nested dict {playerid: {genre: like_rate}}
        self.user_genre_pref = {}
        for _, row in user_genre.iterrows():
            pid = int(row["playerid"])
            genre = row["primary_genre"]
            val = float(row["liked"])
            self.user_genre_pref.setdefault(pid, {})[genre] = val

    def _score_pair(self, pid, gid):
        """
        Compute content-based score for a single (player, game).
        """
        # Base score: how good is this game overall?
        base = self.game_base_score.get(gid, self.global_mean)

        # Game's genre
        genre = self.game_genre.get(gid, None)

        # Player preference for that genre
        if genre is not None:
            player_pref = self.user_genre_pref.get(pid, {}).get(
                genre,
                self.global_genre_like.get(genre, self.global_mean),
            )
        else:
            player_pref = self.global_mean

        # Blend player preference and game base
        return self.w_player * player_pref + self.w_item * base

    def predict(self, X):
        """
        Predict scores for (playerid, gameid) rows in X.
        """
        if self.data_prep is None:
            return np.full(len(X), self.global_mean, dtype=float)

        pids = X["playerid"].astype("Int64").values
        gids = X["gameid"].astype("Int64").values

        scores = np.empty(len(X), dtype=float)
        for i, (pid, gid) in enumerate(zip(pids, gids)):
            scores[i] = self._score_pair(int(pid), int(gid))

        return scores
