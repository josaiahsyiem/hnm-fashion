"""Assemble the ranker's training table: candidates + features + labels."""

import polars as pl

from src.hnm.evaluation.split import load_transactions, train_valid_split
from src.hnm.retrieval.candidates import (
    repurchase_candidates, popularity_candidates, item2item_candidates,
)
from src.hnm.features.build import assemble_training_data


def main():
    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    customers = pl.read_parquet("data/processed/customers.parquet")
    articles = pl.read_parquet("data/processed/articles.parquet")
    print(f"holdout week: {wk}, valid customers: {valid.height:,}")

    valid_customers = valid["customer_id"]
    repurchase = repurchase_candidates(train, recent_weeks=52)
    popular = popularity_candidates(
        train, valid_customers, n=100, recent_weeks=2)
    print("building item2item...")
    i2i = item2item_candidates(train, valid_customers)

    candidates = pl.concat([
        repurchase.select(["customer_id", "article_id"]),
        popular.select(["customer_id", "article_id"]),
        i2i.select(["customer_id", "article_id"]),
    ]).unique()

    candidates = candidates.join(
        valid.select("customer_id"), on="customer_id", how="inner"
    )
    print(f"candidate pairs (scored customers only): {candidates.height:,}")

    data = assemble_training_data(
        candidates, train, customers, articles, valid)

    n_pos = data["label"].sum()
    print(f"\nassembled rows: {data.height:,}")
    print(f"positives: {n_pos:,}  ({n_pos/data.height:.2%})")

    data.write_parquet("data/features/train_table.parquet")
    print("wrote data/features/train_table.parquet")


if __name__ == "__main__":
    main()
