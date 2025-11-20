import pandas as pd
import numpy as np

# Data configuration
PLAYERS_PATH = "xbox_reduced/players_sample.csv"
GAMES_PATH = "xbox_reduced/games.csv"
HISTORY_PATH = "xbox_reduced/history_sample.csv"
ACHIEVEMENTS_PATH = "xbox_reduced/achievements.csv"
PURCHASED_PATH = "xbox_reduced/purchased_games_sample.csv"
PRICES_PATH = "xbox_reduced/prices.csv"
FEATURE_COLS_RAW = ["playerid", "gameid", "completed_ratio"]
MODEL_FEATURE_COLS = ["completed_ratio"]
CF_FEATURE_COLS = ["playerid", "gameid"]  # For collaborative filtering models
HYBRID_FEATURE_COLS = ["playerid", "gameid", "completed_ratio"]  # For hybrid models
TARGET_COL = "liked"
LIKE_THRESH = 0.5
HOLDOUT_POS_PER_USER = 3
RANDOM_SEED = 42
MIN_INTERACTIONS_PER_GAME = 15

def split_data(data):
    data["date_acquired"] = pd.to_datetime(data["date_acquired"])
    train = data[data["date_acquired"].dt.year < 2024]
    test = data[data["date_acquired"].dt.year >= 2024]
    return train, test


class RecommenderDataPrep:
    """Utility class for preparing recommender system data."""

    def __init__(self, player_path: str = PLAYERS_PATH, games_path: str = GAMES_PATH, 
                 history_path: str = HISTORY_PATH, achievements_path: str = ACHIEVEMENTS_PATH, 
                 purchased_path: str = PURCHASED_PATH, prices_path: str = PRICES_PATH,
                 feature_cols: list = None,
                 target_col: str = TARGET_COL, like_thresh: float = LIKE_THRESH,
                 holdout_per_user: int = HOLDOUT_POS_PER_USER, random_seed: int = RANDOM_SEED):
        self.players = player_path
        self.games = games_path
        self.history = history_path
        self.achievements = achievements_path
        self.purchased = purchased_path
        self.prices = prices_path
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
        self.prices = pd.read_csv(self.prices)
        
        self.prices['date_acquired'] = pd.to_datetime(self.prices['date_acquired'])
        
        self.games["primary_genre"] = (self.games["genres"].fillna("").astype(str).str.split(",").str[0].str.strip().replace("", "Unknown"))

        # Player IDs
        self.players["playerid"] = self.players["playerid"].astype("Int64")
        self.history["playerid"] = self.history["playerid"].astype("Int64")

        # Game IDs
        if "gameid" in self.history.columns:
            self.history["gameid"] = self.history["gameid"].astype("Int64")

        if "gameid" in self.achievements.columns:
            self.achievements["gameid"] = self.achievements["gameid"].astype("Int64")

        self.games["gameid"] = self.games["gameid"].astype("Int64")

        self.df = pd.merge(self.players, self.history, on= "playerid", how = "left")
        self.df = pd.merge(self.df, self.achievements[['achievementid', 'gameid']], on= "achievementid", how = "left")
        self.df = pd.merge(self.df, self.games[['gameid', 'title', 'genres', 'primary_genre', 'supported_languages']], on= "gameid", how = "left")
        self.df["gameid"] = self.df["gameid"].astype("Int64")

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

        # Count how many UNIQUE players each game has in TRAIN
        game_player_counts = (
            self.train_df
            .dropna(subset=["gameid"])
            .groupby("gameid")["playerid"]
            .nunique()
        )

        # Keep only games with at least MIN_INTERACTIONS_PER_GAME players
        popular_games = game_player_counts[game_player_counts >= MIN_INTERACTIONS_PER_GAME].index

        print(f"[DEBUG] Keeping {len(popular_games)} games with ≥{MIN_INTERACTIONS_PER_GAME} players")

        # Filter train, test, and full df down to this reduced catalog
        self.train_df = self.train_df[self.train_df["gameid"].isin(popular_games)].copy()
        self.test_df = self.test_df[self.test_df["gameid"].isin(popular_games)].copy()
        self.df = self.df[self.df["gameid"].isin(popular_games)].copy()

        max_train_date = self.train_df["date_acquired"].max()
        # Define cutoff as "one year before the max train date"
        cutoff_date = max_train_date - pd.DateOffset(months=8)
        recent_train = self.train_df[self.train_df["date_acquired"] >= cutoff_date].copy()

        # Genre-level baseline like rates in the recent window
        genre_stats = (recent_train.groupby("primary_genre").agg(
                genre_n_players=("playerid", "nunique"),
                genre_like_rate=("liked", "mean"),))

        game_stats = (recent_train.groupby("gameid").agg(
                n_players=("playerid", "nunique"),
                like_rate=("liked", "mean"),
                primary_genre=("primary_genre", "first"),
            )
        )

        game_stats = game_stats.join(genre_stats[["genre_like_rate"]],on="primary_genre")

        # Popularity score: more players AND higher like rate = higher score.
        # 0.5+0.5*like_rate keeps everything >0 and boosts well-liked games.
        MIN_RECENT_PLAYERS = 10  # try 3–10; start with 5
        game_stats = game_stats[game_stats["n_players"] >= MIN_RECENT_PLAYERS].copy()
        game_stats["pop_score"] = (0.5 * game_stats["n_players"]) * game_stats["like_rate"]

        self.popularity_scores = game_stats["pop_score"].to_dict()

        eval_players = self.test_df["playerid"].unique().tolist()

        all_games = sorted(self.df["gameid"].dropna().astype("Int64").unique().tolist())

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
