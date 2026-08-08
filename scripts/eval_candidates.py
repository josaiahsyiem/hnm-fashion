"""Run candidate generators on the real split and report recall."""

import polars as pl

from src.hnm.evaluation.split import load_transactions, train_valid_split
from src.hnm.retrieval.candidates import (
    repurchase_candidates, popularity_candidates,
    item2item_candidates, candidate_recall,
)


def main():
    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    print(f"holdout week: {wk}, valid customers: {valid.height:,}\n")
    valid_customers = valid["customer_id"]

    repurchase = repurchase_candidates(train, recent_weeks=52)
    popular = popularity_candidates(
        train, valid_customers, n=100, recent_weeks=2)
    print("building item2item (may take a minute)...")
    i2i = item2item_candidates(train, valid_customers)

    for name, c in [("repurchase", repurchase), ("popular-100", popular), ("item2item", i2i)]:
        s = candidate_recall(c, valid)
        print(
            f"{name:<12} recall={s['recall']:.4f}  avg_cands={s['avg_candidates_per_customer']:.1f}")

    combined = pl.concat([
        repurchase.select(["customer_id", "article_id", "method"]),
        popular.select(["customer_id", "article_id", "method"]),
        i2i.select(["customer_id", "article_id", "method"]),
    ]).unique()
    cs = candidate_recall(combined, valid)
    print(f"\nCOMBINED (3 strategies): recall={cs['recall']:.4f}  "
          f"avg_cands={cs['avg_candidates_per_customer']:.1f}")


if __name__ == "__main__":
    main()
