"""Convert raw H&M CSVs into compact parquet files.

Reads the four competition CSVs from data/raw/ and writes memory-efficient
parquet files to data/processed/. The heavy lifting:

- customer_id: 64-char hex string -> Int64 (last 16 hex chars parsed as int).
  This is lossless for this dataset and shrinks the biggest column ~8x.
- article_id: 10-digit code -> Int32.
- t_dat: string -> Date.
- numeric columns downcast to smallest safe dtype.

Run from the repo root:  python -m src.hnm.data.reduce_memory
"""

from pathlib import Path
import polars as pl

RAW = Path("data/raw")
OUT = Path("data/processed")


def hex_id_to_int(col: str) -> pl.Expr:
    """Parse the last 16 hex chars of a hash id column into a UInt64.

    16 hex chars span 0..2^64-1, which overflows signed Int64. We parse
    directly into UInt64 so the parser itself targets the right width.
    Lossless and collision-free for this dataset.
    """
    return (
        pl.col(col)
        .str.slice(-16)
        .str.to_integer(base=16, dtype=pl.UInt64)
        .alias(col)
    )


def convert_transactions() -> None:
    print("transactions: reading...")
    df = pl.read_csv(RAW / "transactions_train.csv")
    df = df.with_columns(
        pl.col("t_dat").str.to_date(),
        hex_id_to_int("customer_id"),
        pl.col("article_id").cast(pl.Int32),
        pl.col("price").cast(pl.Float32),
        pl.col("sales_channel_id").cast(pl.Int8),
    )
    df.write_parquet(OUT / "transactions.parquet")
    print(
        f"transactions: wrote {df.height:,} rows -> {OUT / 'transactions.parquet'}")


def convert_customers() -> None:
    print("customers: reading...")
    df = pl.read_csv(RAW / "customers.csv")
    df = df.with_columns(
        hex_id_to_int("customer_id"),
        pl.col("age").cast(pl.Float32),     # has nulls, keep float
    )
    df.write_parquet(OUT / "customers.parquet")
    print(
        f"customers: wrote {df.height:,} rows -> {OUT / 'customers.parquet'}")


def convert_articles() -> None:
    print("articles: reading...")
    df = pl.read_csv(RAW / "articles.csv")
    df = df.with_columns(
        pl.col("article_id").cast(pl.Int32),
    )
    df.write_parquet(OUT / "articles.parquet")
    print(f"articles: wrote {df.height:,} rows -> {OUT / 'articles.parquet'}")


def convert_sample_submission() -> None:
    print("sample_submission: reading...")
    df = pl.read_csv(RAW / "sample_submission.csv")
    df = df.with_columns(hex_id_to_int("customer_id"))
    df.write_parquet(OUT / "sample_submission.parquet")
    print(
        f"sample_submission: wrote {df.height:,} rows -> {OUT / 'sample_submission.parquet'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    convert_articles()
    convert_customers()
    convert_sample_submission()
    convert_transactions()   # biggest, do last
    print("done.")


if __name__ == "__main__":
    main()
