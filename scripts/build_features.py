"""Assemble the ranker's training table: candidates + features + labels."""

import polars as pl

from src.hnm.evaluation.split import load_transactions, train_valid_split
from src.hnm.retrieval.candidates import repurchase_candidates, popularity_candidates
from src.hnm.features.build import assemble_training_data


def main():
    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    customers = pl.read_parquet("data/processed/customers.parquet")
    articles = pl.read_parquet("data/processed/articles.parquet")
    print(f"holdout week: {wk}, valid customers: {valid.height:,}")

    # Build candidates (the two strategies from Phase 4).
    valid_customers = valid["customer_id"]
    repurchase = repurchase_candidates(train, recent_weeks=52)
    popular = popularity_candidates(
        train, valid_customers, n=100, recent_weeks=2)
    candidates = pl.concat([
        repurchase.select(["customer_id", "article_id"]),
        popular.select(["customer_id", "article_id"]),
    ]).unique()

    # IMPORTANT: only keep candidates for customers we can score (bought in wk 104).
    candidates = candidates.join(
        valid.select("customer_id"), on="customer_id", how="inner"
    )
    print(f"candidate pairs (scored customers only): {candidates.height:,}")

    # Assemble features + labels.
    data = assemble_training_data(
        candidates, train, customers, articles, valid)

    n_pos = data["label"].sum()
    print(f"\nassembled rows: {data.height:,}")
    print(f"positives: {n_pos:,}  ({n_pos/data.height:.2%})")
    print(f"columns ({data.width}): {data.columns}")

    # Save for the ranker.
    data.write_parquet("data/features/train_table.parquet")
    print("\nwrote data/features/train_table.parquet")


if __name__ == "__main__":
    main()
