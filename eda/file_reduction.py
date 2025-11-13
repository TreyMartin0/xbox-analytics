import duckdb
import pandas as pd
from pathlib import Path

# Config
out_dir = Path("xbox_reduced")
players = pd.read_csv('xbox/players.csv')
games = pd.read_csv('xbox/games.csv')
history = pd.read_csv('xbox/history.csv')
purchased_games = pd.read_csv('xbox/purchased_games.csv')
achievements = pd.read_csv('xbox/achievements.csv')
prices = pd.read_csv('xbox/prices.csv')

# Fraction of rows to keep
frac_users = 0.3      # keep 30% of players

# Connect to Duckdb
con = duckdb.connect("xbox_sample.duckdb")

# register dataframes to duckdb
con.register("players", players)
con.register("games", games)
con.register("history", history)
con.register("purchased_games", purchased_games)
con.register("achievements", achievements)
con.register("prices", prices)

# Sample users and games first
con.execute(f"CREATE TABLE users_sample AS SELECT * FROM players USING SAMPLE {frac_users*100} PERCENT;")

# Filter dependent tables
con.execute(f"""
    CREATE OR REPLACE TABLE players_sample AS
    SELECT * FROM players
    WHERE RANDOM() < 0.30
""")

# Keep only entries with remaining playerids 
con.execute("""
    CREATE OR REPLACE TABLE history_sample AS
    SELECT h.*
    FROM history h
    JOIN players_sample u ON h.playerid = u.playerid;
""")

con.execute("""
    CREATE OR REPLACE TABLE purchased_games_sample AS
    SELECT p.*
    FROM purchased_games p
    JOIN players_sample u ON p.playerid = u.playerid;
""")

# Save reduced CSVs 
for table in ["players_sample", "games", "history_sample",
              "purchased_games_sample", "achievements", "prices"]:
    out_path = out_dir / f"{table}.csv"
    con.execute(f"COPY {table} TO '{out_path}' (HEADER, DELIMITER ',');")
    print(f"Saved {table} → {out_path}")


