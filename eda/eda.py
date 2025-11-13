import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import umap
import ast
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import MiniBatchKMeans

#connect 
db_path = Path("xbox_sample.duckdb")
con = duckdb.connect(db_path)

#Load data 
players = pd.read_csv("xbox_reduced/players_sample.csv")
games = pd.read_csv("xbox_reduced/games.csv")
history = pd.read_csv("xbox_reduced/history_sample.csv")
purchases = pd.read_csv("xbox_reduced/purchased_games_sample.csv")
achievements = pd.read_csv("xbox_reduced/achievements.csv")
prices = pd.read_csv("xbox_reduced/prices.csv")

#Summary of data
print("Players:", len(players), players.columns)
print("Games:", len(games), games.columns)
print("History:", len(history), history.columns)
print("Purchases:", len(purchases), purchases.columns)
print("Achievements:", len(achievements), achievements.columns)
print("Prices:", len(prices), prices.columns)

#Missing values summary
print("\nMissing values summary:")
for name, df in {
    "players": players, "games": games, "history": history,
    "purchases": purchases, "achievements": achievements, "prices": prices
}.items():
    print(f"{name:15s}  →  {df.isna().sum().sum()} missing")

#Games per publisher
plt.figure(figsize=(8,4))
games["publishers"].value_counts().head(10).plot(kind="bar", color="steelblue")
plt.title("Publishers by Game Count")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.show()

purchases["games_count"] = (
    purchases["library"]
    .fillna("")
    .astype(str)
    .str.count(r"\d+")
)

# Genre popularity distribution
plt.figure(figsize=(10,5))
sns.countplot(y="genres", data=games, order=games["genres"].value_counts().index[:10])
plt.title("Top 10 Game Genres on Xbox")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()


# Quick stats about game ownership
mean_gc = purchases["games_count"].mean()
median_gc = purchases["games_count"].median()
p90 = purchases["games_count"].quantile(0.90)
p99 = purchases["games_count"].quantile(0.99)

all_players = set(players["playerid"])
players_with_games = set(purchases["playerid"])

# Compute difference
players_without_games = all_players - players_with_games

print(f"Total players: {len(all_players):,}")
print(f"Players without any games: {len(players_without_games):,}")
print(f"Percentage without games: {len(players_without_games) / len(all_players) * 100:.2f}%")

print(f"Players who own a game: {purchases['playerid'].nunique():,}")
print(f"Avg games/player: {mean_gc:.2f} | Median: {median_gc:.0f} | P90: {p90:.0f} | P99: {p99:.0f}")
print(f"Max games owned by a player: {purchases['games_count'].max()}")

# (A) Main distribution (handles skew)
plt.figure(figsize=(8,4))
sns.histplot(purchases["games_count"], kde=True)
plt.title("Distribution of Number of Games per Player")
plt.xlabel("Games Owned")
plt.ylabel("Players")
plt.tight_layout()
plt.show()

# Zoom into the bulk
clip_cap = int(p99)  # clip to the 99th percentile
plt.figure(figsize=(8,4))
sns.histplot(purchases["games_count"].clip(upper=clip_cap), bins=50)
plt.title(f"Games per Player (clipped at {clip_cap})")
plt.xlabel("Games Owned (clipped)")
plt.ylabel("Players")
plt.tight_layout()
plt.show()

#Distribution of Achievements Earned per Player
achievements_per_player = history.groupby("playerid")["achievementid"].nunique().reset_index()
plt.figure(figsize=(8,4))
sns.histplot(achievements_per_player["achievementid"], kde=True)
plt.title("Distribution of Achievements Earned per Player")
plt.xlabel("Unique Achievements Earned")
plt.ylabel("Players")
plt.tight_layout()
plt.show()


player_game = pd.merge(
    history[["playerid", "achievementid"]],
    achievements[["achievementid", "gameid"]],
    on="achievementid",
    how="left"
)

# Count unique players per game
game_popularity = (
    player_game.groupby("gameid")["playerid"]
    .nunique()
    .reset_index(name="num_players")
)

avg_price = prices.groupby("gameid")["usd"].mean().reset_index()

merged = pd.merge(game_popularity, avg_price, on="gameid", how="inner")

# Create clean labeled bins
bins = range(0, 101, 10)
labels = [f"${bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
merged["price_bin"] = pd.cut(
    merged["usd"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Aggregate
agg = merged.groupby("price_bin", observed=True)["num_players"].mean().reset_index()

# Plot
plt.figure(figsize=(8,4))
sns.barplot(x="price_bin", y="num_players", data=agg, color="cornflowerblue")
plt.title("Average Game Popularity by Price Range")
plt.xlabel("Price Range (USD)")
plt.ylabel("Average Unique Players")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


player_game = (
    history.merge(achievements[["achievementid", "gameid"]], on="achievementid", how="left")
            .merge(games[["gameid", "title"]], on="gameid", how="left")
)

# Count unique players per game
popularity = (
    player_game.groupby(["gameid", "title"])["playerid"]
    .nunique()
    .reset_index(name="num_players")
    .sort_values("num_players", ascending=False)
)

# Select top 20 most popular
top20 = popularity.head(20)

# Plot nicely
plt.figure(figsize=(10,6))
sns.barplot(y="title", x="num_players", data=top20, color="cornflowerblue")
plt.title("Top 20 Most Popular Games by Unique Players")
plt.xlabel("Unique Players")
plt.ylabel("Game Title")
plt.tight_layout()
plt.show()

# top 10 stats
print(top20.head(10))



ach_counts = (
    history.groupby("achievementid")["playerid"]
    .nunique()
    .reset_index(name="num_players")
)

# Merge to get achievement details
ach_info = pd.merge(
    ach_counts,
    achievements[["achievementid", "gameid", "title", "points"]],
    on="achievementid",
    how="left"
)

# Merge again to get the game title
ach_info = pd.merge(
    ach_info,
    games[["gameid", "title"]].rename(columns={"title": "game_title"}),
    on="gameid",
    how="left"
)

# Get top 10
top10 = ach_info.sort_values("num_players", ascending=False).dropna(subset=["game_title"]).head(10)

plt.figure(figsize=(10,6))
sns.set_theme(style="whitegrid")

# Sort by number of players for consistent order
top10 = top10.sort_values("num_players", ascending=True)

# Use a color palette that transitions by game popularity
palette = sns.color_palette("Spectral", n_colors=len(top10["game_title"].unique()))

barplot = sns.barplot(
    x="num_players",
    y="title",
    data=top10,
    hue="game_title",
    dodge=False,
    palette=palette,
    edgecolor="black"
)

plt.figure(figsize=(10,6))
sns.set_theme(style="whitegrid")

#top 10 achieved xbox achievements
top10 = top10.sort_values("num_players", ascending=True)
palette = sns.color_palette("tab10", n_colors=len(top10["game_title"].unique()))

barplot = sns.barplot(
    x="num_players",
    y="title",
    data=top10,
    hue="game_title",
    dodge=False,
    palette=palette,
    edgecolor="black"
)

plt.title("Top 10 Most Achieved Xbox Achievements", fontsize=15, weight="bold", pad=15)
plt.xlabel("Number of Players Who Achieved", fontsize=12)
plt.ylabel("Achievement Title", fontsize=12)

# Add labels on bars
for container in barplot.containers:
    barplot.bar_label(container, fmt="%d", label_type="edge", padding=3, fontsize=9, color="black")

plt.legend(
    title="Game Title",
    title_fontsize=10,
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12), 
    ncol=3,                  
    frameon=True,
    shadow=False
)

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

history["date_acquired"] = pd.to_datetime(history["date_acquired"])

# Group by month
monthly = (
    history.groupby(pd.Grouper(key="date_acquired", freq="M"))["playerid"]
    .nunique()
    .reset_index(name="unique_players")
)

highlight = monthly.iloc[monthly["unique_players"].idxmax()]
monthly = monthly.sort_values("date_acquired").copy()

# Find peak for annotation
peak_idx = monthly["unique_players"].idxmax()
peak = monthly.loc[peak_idx]

# Plot
plt.figure(figsize=(11, 6))
ax = plt.gca()

# main line
ax.plot(
    monthly["date_acquired"], monthly["unique_players"],
    lw=2.5, color="black", marker="o", ms=4, alpha=0.9, label="Monthly players"
)

# highlight peak
ax.scatter(peak["date_acquired"], peak["unique_players"], s=80, color="#F4A261", zorder=5)
ax.annotate(
    f"Peak: {int(peak['unique_players']):,}",
    xy=(peak["date_acquired"], peak["unique_players"]),
    xytext=(15, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="#F4A261"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F4A261", alpha=0.9),
    color="#F4A261", weight="bold"
)

# Axis formatting
ax.set_title("Xbox Player Engagement Over Time", fontsize=18, weight="bold", pad=12)
ax.set_ylabel("Unique Active Players", fontsize=12)

# y-axis with thousands separators
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

# date tickss
ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3,6,9,12]))

# subtle grid
ax.grid(True, axis="y", which="major", alpha=0.25)
ax.grid(True, axis="y", which="minor", alpha=0.10)
ax.set_axisbelow(True)

# clean spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# legend below the plot
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

plt.tight_layout()
plt.show()

RANDOM_STATE= 42
MIN_OWNERS_PER_GAME = 5 
N_CLUSTERS = 6 
N_NEIGHBORS = 20
MIN_DIST = 0.1 

games["gameid"] = games["gameid"].astype(str)

# Build interaction matrix
purchases["library"] = purchases["library"].apply(ast.literal_eval)

# explode to (playerid, gameid) rows
purchases_expanded = purchases.explode("library").rename(columns={"library": "gameid"})
purchases_expanded["gameid"] = purchases_expanded["gameid"].astype(str)

# filter out ultra-rare games
owners_per_game = purchases_expanded.groupby("gameid")["playerid"].nunique()
keep_games = owners_per_game[owners_per_game >= MIN_OWNERS_PER_GAME].index
purchases_expanded = purchases_expanded[purchases_expanded["gameid"].isin(keep_games)]

# player x game binary matrix
matrix = pd.crosstab(
    purchases_expanded["playerid"],
    purchases_expanded["gameid"]
)

# L2-normalize rows so whales don't dominate
matrix_norm = normalize(matrix, norm="l2")  

# UMAP (on games)
# transpose -> games x players
reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    metric="cosine",
    random_state=RANDOM_STATE
)
embedding = reducer.fit_transform(matrix_norm.T)

# dataframe of embedded games
umap_df = pd.DataFrame(embedding, columns=["x", "y"])
umap_df["gameid"] = matrix.columns.astype(str)

# attach titles/genres
umap_df = umap_df.merge(games[["gameid", "title", "genres"]], on="gameid", how="left")

#parse genres into list
def parse_genres(g):
    if pd.isna(g): 
        return []
    if isinstance(g, str) and g.startswith("["):
        try:
            out = ast.literal_eval(g)
            return [str(s).strip() for s in out]
        except Exception:
            pass
    if isinstance(g, str):
        return [s.strip() for s in g.replace("|", ",").split(",") if s.strip()]
    return []

umap_df["genre_list"] = umap_df["genres"].apply(parse_genres)

# CLUSTER IN 2-D
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, batch_size=2048)
umap_df["cluster"] = km.fit_predict(umap_df[["x", "y"]])

# cluster summary: size, top 3 genres, example titles
rows = []
for c, grp in umap_df.groupby("cluster"):
    all_genres = pd.Series([g for lst in grp["genre_list"] for g in lst])
    top_genres = ", ".join(all_genres.value_counts().head(3).index.tolist()) or "—"
    sample_titles = ", ".join(grp["title"].dropna().head(3).tolist())
    rows.append(dict(cluster=c, n=len(grp), top_genres=top_genres, example_titles=sample_titles))
cluster_summary = pd.DataFrame(rows).sort_values("n", ascending=False)
print("\n=== Cluster Summary ===")
print(cluster_summary.to_string(index=False))

sns.set_context("talk")
sns.set_style("whitegrid")

# Plot with cluster labels (top genres) at centroids
centroids = umap_df.groupby("cluster")[["x", "y"]].median().reset_index()
labels_map = cluster_summary.set_index("cluster")["top_genres"]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=umap_df, x="x", y="y", hue="cluster",
    palette=sns.color_palette("tab20", n_colors=N_CLUSTERS), s=14, linewidth=0, alpha=0.7, legend=False
)
for _, row in centroids.iterrows():
    c = int(row["cluster"])
    txt = f"C{c}: {labels_map.get(c, '—')}"
    plt.text(row["x"], row["y"], txt, fontsize=9, weight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.75))
plt.title("UMAP with cluster labels (top genres)", weight="bold")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.tight_layout(); plt.show()