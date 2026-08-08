"""Tie split + metric together, and compute a popularity baseline.

The baseline: predict the 12 most popular articles from the training weeks
for EVERY customer. It ignores personalization entirely, so it's the floor
your real model must beat.
"""

import polars as pl

from src.hnm.evaluation.split import load_transactions, train_valid_split
from src.hnm.evaluation.metric import mapk


def most_popular(train: pl.DataFrame, n: int = 12, recent_weeks: int = 1) -> list[int]:
    """Top-n article_ids by purchase count in the last `recent_weeks` of train."""
    last = train["week"].max()
    pool = train.filter(pl.col("week") > last - recent_weeks)
    top = (
        pool.group_by("article_id").len()
        .sort("len", descending=True)
        .head(n)["article_id"]
        .to_list()
    )
    return top


def evaluate_predictions(valid: pl.DataFrame, pred_map: dict[int, list[int]]) -> float:
    """Score a {customer_id: [article_ids]} dict against validation ground truth."""
    actuals, preds = [], []
    for row in valid.iter_rows(named=True):
        cid = row["customer_id"]
        actuals.append(row["actual"])
        preds.append(pred_map.get(cid, []))
    return mapk(actuals, preds, k=12)


def popularity_baseline() -> float:
    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    top12 = most_popular(train, n=12, recent_weeks=1)
    # Same 12 items predicted for every validation customer.
    pred_map = {cid: top12 for cid in valid["customer_id"].to_list()}
    score = evaluate_predictions(valid, pred_map)
    print(f"holdout week: {wk}")
    print(f"top-12 popular (last train week): {top12}")
    print(f"POPULARITY BASELINE MAP@12: {score:.5f}")
    return score


if __name__ == "__main__":
    popularity_baseline()
