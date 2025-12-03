Xbox Analytics — Game Recommendation System
_A data-driven recommender built from real Xbox player engagement behavior._

Overview
This repository contains an end-to-end analytics and recommendation pipeline using Xbox player data.
It includes feature engineering, exploratory data analysis, UMAP visualizations, multiple recommendation models, full evaluation scripts, and a polished final report.
This repo demonstrates real-world engagement modeling, collaborative filtering, and hybrid recommender techniques.

Core Capabilities
Full Data Engineering Pipeline
- Merges achievements, ownership history, timestamps, and game metadata
- Computes behavior-driven features (completed ratio, recency, velocity)
- Reduces raw sample → structured analytics dataset (xbox_sample.duckdb)

Multiple Recommender Models
Inside models/:
- random_rec.py — baseline
- content_rec.py — feature-based
- cf_rec.py — collaborative filtering
- rf_rec.py — RandomForest baseline
- hybrid_rec.py — CF + Popularity + Content blended recommender

Complete Evaluation Framework
Inside evaluator.py:
- Precision@K
- Recall@K
- MAP@K
- Ranking curve visualizations
- Confusion matrices
- User-level recommendation summaries

Visualizations & EDA
Inside eda/:
- Distribution of achievements
- Distribution of games owned
- Temporal engagement trends
- UMAP cluster embeddings
- Genre frequency summaries

Repository Structure
xbox-analytics/
│
├── eda/
│   ├── Graphs/                  # All analysis figures (PNG)
│   │   ├── achievements_dist.png
│   │   ├── games_owned_dist.png
│   │   ├── umap_clusters.png
│   │   └── ...
│   ├── eda_summary.md           # Markdown writeup of EDA
│   ├── eda.py                   # EDA generation scripts
│   ├── file_reduction.py        # Preprocessing / downsampling
│   └── xbox_sample.duckdb       # Reduced + optimized DB
│
├── models/                      # All recommender implementations
│   ├── base_rec.py              # Base abstract model
│   ├── random_rec.py
│   ├── content_rec.py
│   ├── cf_rec.py
│   ├── rf_rec.py
│   └── hybrid_rec.py
│
├── xbox_reduced/                # Cleaned CSV dataset (30% sample)
│   ├── achievements.csv
│   ├── games.csv
│   ├── history_sample.csv
│   ├── players_sample.csv
│   ├── prices.csv
│   └── purchased_games_sample.csv
│
├── build_data.py                # Feature construction + data merging
├── evaluator.py                 # Core evaluation script
├── ranking_curve.png            # Precision-Recall curves
├── confusion_matrix_cf.png      # CF confusion matrix
│
├── final_report.md              # Polished 3-page whitepaper
├── report1.md                   # Draft report (deprecated)
│
└── README.md                    # You are here

How the Recommender Works
Feature Engineering
Each player–game interaction is mapped to meaningful behavioral signals:

- Completed Ratio: % of total achievements earned per game
- Recency Weight: exp(-days_since_last_achievement / 180)
- Velocity: achievements earned per hour

These are combined into a final engagement score:

score = 0.60 * completed_ratio +
        0.30 * recency_weight   +
        0.10 * velocity_norm

Games with ≥ 15 unique players are retained to avoid sparsity issues.

Contact:
James (Trey) Martin
Email: treymartin.wv@gmail.com

