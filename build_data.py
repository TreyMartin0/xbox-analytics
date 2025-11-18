import pandas as pd
import numpy as np

# Data configuration
PLAYERS_PATH = "xbox_reduced/players_sample.csv"
GAMES_PATH = "xbox_reduced/games.csv"
HISTORY_PATH = "xbox_reduced/history_sample.csv"
ACHIEVEMENTS_PATH = "xbox_reduced/achievements.csv"
PURCHASED_PATH = "xbox_reduced/purchased_games_sample.csv"
FEATURE_COLS_RAW = ["playerid", "gameid", "completed_ratio"]
MODEL_FEATURE_COLS = ["completed_ratio"]
CF_FEATURE_COLS = ["playerid", "gameid"]  # For collaborative filtering models
HYBRID_FEATURE_COLS = ["playerid", "gameid", "completed_ratio"]  # For hybrid models
TARGET_COL = "liked"
LIKE_THRESH = 0.4
HOLDOUT_POS_PER_USER = 3
RANDOM_SEED = 42

def split_data(data):
    data["date_acquired"] = pd.to_datetime(data["date_acquired"])
    train = data[data["date_acquired"].dt.year < 2023]
    test = data[data["date_acquired"].dt.year >= 2023]
    return train, test


class RecommenderDataPrep:
    """Utility class for preparing recommender system data."""

    def __init__(self, player_path: str = PLAYERS_PATH, games_path: str = GAMES_PATH, 
                 history_path: str = HISTORY_PATH, achievements_path: str = ACHIEVEMENTS_PATH, 
                 purchased_path: str = PURCHASED_PATH, feature_cols: list = None,
                 target_col: str = TARGET_COL, like_thresh: float = LIKE_THRESH,
                 holdout_per_user: int = HOLDOUT_POS_PER_USER, random_seed: int = RANDOM_SEED):
        self.players = player_path
        self.games = games_path
        self.history = history_path
        self.achievements = achievements_path
        self.purchased = purchased_path
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS_RAW
        self.target_col = target_col
        self.like_thresh = like_thresh
        self.holdout_per_user = holdout_per_user
        self.random_seed = random_seed

        self.df = None
        self.train_df = None
        self.test_df = None
        self.eval_users = None
        self.candidates_by_ply = None
        self.relevant_by_ply = None
        self.player_features = None
        self.game_features = None
        self.player_info = None

    def load_and_prepare(self):
        """Load data and prepare train/test splits."""
        # Load data
        self.players = pd.read_csv(self.players)
        self.games = pd.read_csv(self.games)
        self.history = pd.read_csv(self.history)
        self.achievements = pd.read_csv(self.achievements)
        self.purchased = pd.read_csv(self.purchased)

        self.df = pd.merge(self.players, self.history, on= "playerid", how = "left")
        self.df = pd.merge(self.df, self.achievements[['achievementid', 'gameid']], on= "achievementid", how = "left")
        self.df = pd.merge(self.df, self.games[['gameid', 'title', 'genres', 'supported_languages']], on= "gameid", how = "left")

        # Per-player per-game stats (completed + last_date)
        pg_stats = (self.df.dropna(subset=["achievementid"]).groupby(["playerid", "gameid"]).agg(completed=("achievementid", "nunique"), last_date_acquired=("date_acquired", "max")).reset_index())
        # Total achievements per game
        game_totals = (self.df.dropna(subset=["achievementid"]).groupby("gameid")["achievementid"].nunique().rename("total_ach_count").reset_index())

        # Merge stats back onto self.df
        self.df = self.df.merge(pg_stats, on=["playerid", "gameid"], how="left")
        self.df = self.df.merge(game_totals, on="gameid", how="left")
        # Ratio
        self.df["completed_ratio"] = self.df["completed"] / self.df["total_ach_count"]
        
        # Keep only rows with a real achievement
        df_nonnull = self.df[self.df["achievementid"].notna()].copy()

        # Sort so the last achievement per (player, game) is at the bottom
        df_nonnull = df_nonnull.sort_values(
            ["playerid", "gameid", "date_acquired"]
        )

        # Drop duplicates, keeping only the last achievement row per (player, game)
        df_latest = df_nonnull.drop_duplicates(
            ["playerid", "gameid"],
            keep="last"
        ).copy()

        df_latest.loc[:, "date_acquired"] = df_latest["last_date_acquired"]
        df_latest = df_latest.drop(columns=["last_date_acquired"])

        # This is now your per-player-per-game, latest-achievement table
        self.df = df_latest.reset_index(drop=True)

        # liked = 1 if completion ratio >= threshold
        self.df["liked"] = (self.df["completed_ratio"] >= self.like_thresh).astype(int)

        self.train_df, self.test_df = split_data(self.df)

        eval_players = self.test_df["playerid"].unique().tolist()

        all_games = sorted(self.df["gameid"].unique().tolist())

        # Games seen in training per player
        seen_train_by_player = {pid: set(g["gameid"].tolist())
            for pid, g in self.train_df.groupby("playerid")
        }

        # Candidates: games not seen in training
        candidates_by_player = {pid: [gid for gid in all_games
            if gid not in seen_train_by_player.get(pid, set())]
            for pid in eval_players
        }

        # Relevant positives in test 
        relevant_by_player = {}
        for pid, g in self.test_df.groupby("playerid"):
            liked_games = set(g.loc[g["liked"] == 1, "gameid"].tolist())
            cand_set = set(candidates_by_player.get(pid, []))
            rel = liked_games & cand_set
            if rel:
                relevant_by_player[pid] = rel

        # Final eval users only those with at least 1 relevant item
        self.eval_users = sorted(relevant_by_player.keys())
        self.candidates_by_ply = {pid: candidates_by_player[pid] for pid in self.eval_users}
        self.relevant_by_ply = relevant_by_player

        self.player_features = self.train_df.groupby("playerid")[["completed_ratio"]].mean()
        self.game_features = self.df.groupby("gameid")[["completed_ratio"]].mean()
        self.player_info = self.df.groupby("playerid")[["nickname"]].first()

    def get_training_data(self, model_feature_cols: list):
        """Get training features and target."""
        X_train = self.train_df[model_feature_cols].astype(float)
        y_train = self.train_df[self.target_col].astype(float)
        return X_train, y_train

    def _create_cf_interactions(self, pid, candidates):
        """Create feature matrix for collaborative filtering (IDs only)."""
        n = len(candidates)
        X_cand = pd.DataFrame({
            "playerid": [pid] * n,
            "gameid": candidates
        })
        return X_cand, candidates

    def _create_content_interactions(self, candidates, player_feat):
        """Create feature matrix for content-based models (features only)."""
        game_feats = self.game_features.loc[candidates][["completed_ratio"]].values.astype(float)
        n = len(candidates)

        X_cand = pd.DataFrame(
            game_feats,
            columns=["completed_ratio"]
        )
        return X_cand, candidates

    def _create_hybrid_interactions(self, pid, candidates, player_feat):
        """Create feature matrix for hybrid models (IDs + features)."""
        game_feats = self.game_features.loc[candidates][["completed_ratio"]].values.astype(float)
        n = len(candidates)

        X_cand = pd.DataFrame(
            np.column_stack([
                np.full(n, float(pid)),   # player id as numeric
                np.array(candidates, dtype=float),
                game_feats
            ]),
            columns=["playerid", "gameid", "completed_ratio"]
        )
        return X_cand, candidates

    def create_unseen_interactions(self, pid, model_feature_cols: list):
        """Create feature matrix for unseen interactions for a given adventurer."""
        candidates = self.candidates_by_ply[pid]

        # CF model: only needs playerid & gameid
        if set(model_feature_cols) == {"playerid", "gameid"}:
            return self._create_cf_interactions(pid, candidates)

        if "completed_ratio" in model_feature_cols:
            # Lookup per-game completed_ratio, fill missing with global mean
            global_mean = float(self.df["completed_ratio"].mean())
            comp = (
                self.game_features.reindex(candidates)["completed_ratio"]
                .fillna(global_mean)
                .values
            )
            X_cand = pd.DataFrame({"completed_ratio": comp})
        else:
            # Fallback: just give a dummy column if some other features were requested
            X_cand = pd.DataFrame(
                np.zeros((len(candidates), len(model_feature_cols))),
                columns=model_feature_cols,
            )

        return X_cand, candidates

    def get_positive_rate(self):
        """Calculate global positive rate."""
        return self.df["liked"].mean()

    def get_user_count(self):
        """Get total number of unique users."""
        return self.df["playerid"].nunique()
