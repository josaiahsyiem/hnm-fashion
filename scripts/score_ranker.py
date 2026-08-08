"""Score the trained ranker with MAP@12 on the held-out week."""

import joblib
import numpy as np
import polars as pl

from src.hnm.ranking.train import FEATURES
from src.hnm.evaluation.split import load_transactions, train_valid_split
from src.hnm.evaluation.evaluate import evaluate_predictions


def main():
    data = pl.read_parquet("data/features/train_table.parquet")
    model = joblib.load("models/ranker.pkl")

    # Predict a score for every candidate.
    X = data.select(FEATURES).to_numpy()
    scores = model.predict(X)
    data = data.with_columns(pl.Series("score", scores))

    # Top-12 articles per customer by score.
    ranked = (
        data.sort(["customer_id", "score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("preds"))
    )
    pred_map = {row["customer_id"]: row["preds"]
                for row in ranked.iter_rows(named=True)}

    # Ground truth + score.
    tx = load_transactions()
    _, valid, wk = train_valid_split(tx)
    score = evaluate_predictions(valid, pred_map)

    print(f"holdout week: {wk}")
    print(f"RANKER MAP@12: {score:.5f}")
    print(f"(popularity baseline was 0.00900)")


if __name__ == "__main__":
    main()
