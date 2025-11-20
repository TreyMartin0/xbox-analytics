import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models import (
    RandomRecommender,
    CFRecommender,
    PopularityRecommender
)
from build_data import RecommenderDataPrep, MODEL_FEATURE_COLS, CF_FEATURE_COLS, HYBRID_FEATURE_COLS

# Evaluation configuration
TOP_K = 5
SAMPLE_N = 5


def precision_at_k(recommended, relevant, k):
    """Calculate precision at k."""
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Calculate recall at k."""
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """Calculate average precision at k."""
    ap, hits = 0.0, 0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k)


def evaluate_rankings(rec_lists, relevant_sets, k):
    """Evaluate recommendation rankings across all users."""
    p_list, r_list, ap_list = [], [], []
    for aid, recs in rec_lists.items():
        rel = relevant_sets[aid]
        p_list.append(precision_at_k(recs, rel, k))
        r_list.append(recall_at_k(recs, rel, k))
        ap_list.append(average_precision_at_k(recs, rel, k))
    return (
        np.mean(p_list),
        np.mean(r_list),
        np.mean(ap_list),
    )


def print_results(results, data_prep, eval_users):
    """Print evaluation results in a formatted table."""
    pos_rate = data_prep.get_positive_rate()
    user_count = data_prep.get_user_count()

    print("\n=== Recommender Comparison (TOP_K = {}) ===".format(TOP_K))
    print(f"Users total: {user_count} | Evaluated (with ≥1 positive): {len(eval_users)}")
    print(f"Global positive rate (liked): {pos_rate:.3f}\n")

    for name, metrics in results.items():
        print(f"{name} recommender:")
        print(f"  Precision@K: {metrics['precision']:.4f}")
        print(f"  Recall@K:    {metrics['recall']:.4f}")
        print(f"  MAP@K:       {metrics['map']:.4f}")
        print()


def print_sample_recommendations(recommendations, data_prep, eval_users):
    """Print sample recommendations for first N users in a more readable format."""
    # Lookup for game info (title, genres) by gameid
    game_info = (
        data_prep.df
        .groupby("gameid")[["title", "genres"]]
        .first()
    )

    def fmt_game(gid):
        """Format a game as '1234 (Game Title)' without .0 and with safe fallbacks."""
        # Many IDs are floats (e.g., 2363.0); normalize to int when possible
        try:
            gid_int = int(gid)
        except (ValueError, TypeError):
            gid_int = gid

        row = None
        if gid in game_info.index:
            row = game_info.loc[gid]
        elif gid_int in game_info.index:
            row = game_info.loc[gid_int]

        if row is None:
            return f"{gid_int}"

        title = row["title"]
        title_str = "Unknown title" if pd.isna(title) else str(title)
        return f"{gid_int} ({title_str})"

    def format_list(items, max_items=5):
        """Pretty-print a list of strings, truncating if it’s long."""
        items = list(items)
        if not items:
            return "None"
        if len(items) <= max_items:
            return ", ".join(items)
        shown = ", ".join(items[:max_items])
        return f"{shown}, ... (+{len(items) - max_items} more)"

    print(
        f"\n--- Sample top-{TOP_K} recommendations "
        f"(first {SAMPLE_N} evaluated players) ---"
    )

    sample_players = eval_users[:SAMPLE_N]

    for pid in sample_players:
        rel = sorted(data_prep.relevant_by_ply.get(pid, set()))
        # adv_info: indexed by playerid, with "nickname" column
        nickname = str(data_prep.player_info.loc[pid, "nickname"])

        print(f"\nPlayer {pid} ({nickname}):")

        # Held-out positives
        if rel:
            held_list = [fmt_game(gid) for gid in rel]
            print(f"  Held-out positives in test:")
            print(f"    {format_list(held_list, max_items=8)}")
        else:
            print("  [WARN] No held-out positives found in candidates (unexpected).")

        # Model-wise recommendations
        for name in recommendations.keys():
            top_recs = recommendations[name].get(pid, [])[:TOP_K]
            hits = [gid for gid in top_recs if gid in rel]
            formatted_recs = [fmt_game(gid) for gid in top_recs]

            print(f"  {name:20s} top-{TOP_K}:")
            print(f"    {format_list(formatted_recs, max_items=TOP_K)}")
            print(f"    hits: {len(hits)}")

        cand = data_prep.candidates_by_ply.get(pid, [])
        print(f"  Candidate set size: {len(cand)}")

def plot_cf_confusion_matrix(data_prep, cf_model, threshold=0.5):
    """
    Build a confusion matrix for the CollaborativeFiltering model
    on the held-out test interactions.

    We treat each (player, game) pair in test_df as a binary
    classification: liked (1) vs not liked (0), using the CF
    score as a probability and thresholding at `threshold`.
    """
    # Use only rows with valid playerid and gameid
    test = data_prep.test_df.dropna(subset=["playerid", "gameid"]).copy()

    # Features for CF model (IDs only)
    X_test = test[CF_FEATURE_COLS].astype(float)
    y_true = test["liked"].astype(int).values

    # Predicted scores from CF model
    y_scores = cf_model.predict(X_test)

    # Turn scores into binary predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix – Collaborative Filtering")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0 (not liked)", "Pred 1 (liked)"], rotation=25, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])

    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("confusion_matrix_cf.png", dpi=200)
    # plt.show()  # Uncomment when running locally

    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")



def main():
    # Prepare data
    print("Loading and preparing data...")
    data_prep = RecommenderDataPrep()
    data_prep.load_and_prepare()

    # Initialize models with default hyperparameters
    # Content-based models (use features)
    content_models = {
        "Random": (RandomRecommender(), MODEL_FEATURE_COLS),
    }

    # Collaborative filtering models (only use IDs)
    cf_models = {
        "CollaborativeFiltering": (CFRecommender(), CF_FEATURE_COLS),
        "Popularity" : (PopularityRecommender(popularity_scores=data_prep.popularity_scores), CF_FEATURE_COLS)
    }

    # if SURPRISE_AVAILABLE:
    #     cf_models["Surprise"] = (SurpriseRecommender(), CF_FEATURE_COLS)

    # Hybrid model (uses both CF and content features)
    # hybrid_models = {
    #     "Hybrid": (HybridRecommender(), HYBRID_FEATURE_COLS)
    # }

    # Combine all models
    all_models = {**content_models, **cf_models} # **hybrid_models}

    # Train all models
    for name, (model, feature_cols) in all_models.items():
        print(f"Training {name}...")
        X_train, y_train = data_prep.get_training_data(feature_cols)
        model.fit(X_train, y_train)

    cf_model = all_models["CollaborativeFiltering"][0]

    # Generate recommendations
    recommendations = {}
    for name, (model, feature_cols) in all_models.items():
        print(f"Generating recommendations for {name}...")
        recommendations[name] = {}
        for pid in data_prep.eval_users:
            X_cand, cand_ids = data_prep.create_unseen_interactions(pid, feature_cols)
            recommendations[name][pid] = model.recommend(X_cand, cand_ids)

    # Evaluate all models
    results = {}
    for name in all_models.keys():
        p, r, map_score = evaluate_rankings(recommendations[name], data_prep.relevant_by_ply, TOP_K)
        results[name] = {"precision": p, "recall": r, "map": map_score}

    # Print results
    print_results(results, data_prep, data_prep.eval_users)
    print_sample_recommendations(recommendations, data_prep, data_prep.eval_users)
    plot_cf_confusion_matrix(data_prep, cf_model, threshold=0.5)


if __name__ == "__main__":
    main()