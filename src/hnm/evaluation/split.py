"""Time-based train/validation split for the H&M recommender.

The competition asks us to predict each customer's purchases in the 7 days
immediately after the training period. We mimic that locally: hold out the
last full week as validation, train on everything strictly before it.
"""

from pathlib import Path
import polars as pl


def _repo_root() -> Path:
    here = Path.cwd()
    return here if (here / "data").exists() else here.parent


def load_transactions() -> pl.DataFrame:
    return pl.read_parquet(_repo_root() / "data" / "processed" / "transactions.parquet")


def add_week_index(tx: pl.DataFrame) -> pl.DataFrame:
    """Add a 0-based 'week' column (weeks since the first transaction date)."""
    first_day = tx["t_dat"].min()
    return tx.with_columns(
        ((pl.col("t_dat") - first_day).dt.total_days() //
         7).cast(pl.Int32).alias("week")
    )


def train_valid_split(tx: pl.DataFrame, valid_week: int | None = None):
    """Split into (train, valid_ground_truth).

    train:  all transactions with week < valid_week.
    valid:  ground truth for the held-out week — one row per customer with
            the list of article_ids they actually bought that week.

    If valid_week is None, uses the last week present in the data.
    """
    tx = add_week_index(tx)
    if valid_week is None:
        valid_week = tx["week"].max()

    train = tx.filter(pl.col("week") < valid_week)

    valid = (
        tx.filter(pl.col("week") == valid_week)
        .group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("actual"))
    )
    return train, valid, valid_week


if __name__ == "__main__":
    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    print(f"holdout week: {wk}")
    print(f"train rows:   {train.height:,}")
    print(f"valid customers (bought something that week): {valid.height:,}")
    print(valid.head())
