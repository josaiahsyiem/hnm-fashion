"""Phase 6 — LightGBM ranker.

Trains an LGBMRanker on the assembled candidate table. The ranker optimizes
the order of candidates WITHIN each customer (what MAP@12 rewards), so the
data must be grouped by customer and we pass group sizes to the model.
"""

import numpy as np
import polars as pl
import lightgbm as lgb

FEATURES = [
    "art_total_buys", "art_unique_buyers", "art_mean_price",
    "art_weeks_since_last_sold", "art_recent_buys",
    "cust_total_buys", "cust_unique_articles", "cust_mean_price",
    "cust_mean_channel", "cust_weeks_since_last_buy", "cust_age",
    "cust_art_prior_buys", "cust_group_prior_buys",
]


def train_ranker(table_path: str = "data/features/train_table.parquet") -> lgb.LGBMRanker:
    data = pl.read_parquet(table_path)

    # CRITICAL: sort by customer so each customer's rows are contiguous,
    # then compute group sizes (rows per customer) in that same order.
    data = data.sort("customer_id")
    group_sizes = data.group_by("customer_id", maintain_order=True).len()[
        "len"].to_list()

    # Sanity: group sizes must sum to total rows.
    assert sum(group_sizes) == data.height, "group sizes do not cover all rows"

    X = data.select(FEATURES).to_numpy()
    y = data["label"].to_numpy()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X, y,
        group=group_sizes,
        eval_at=[12],
    )

    # Report feature importances — a quick reality check on what drives ranking.
    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    print("\nfeature importances (gain-split count):")
    for name, imp in importances:
        print(f"  {name:<28} {imp}")

    return model


if __name__ == "__main__":
    import joblib
    model = train_ranker()
    joblib.dump(model, "models/ranker.pkl")
    print("\nsaved models/ranker.pkl")
