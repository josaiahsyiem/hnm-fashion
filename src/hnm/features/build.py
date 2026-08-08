"""Feature engineering for the ranking stage.

Features attach to each (customer, article) candidate pair so the ranker
can score them. Three groups: article-level, customer-level, and
interaction-level. This first pass builds a small, high-signal set — the
EDA flagged popularity/recency, age, and channel as the load-bearing ones.
"""

import polars as pl


def article_features(train: pl.DataFrame) -> pl.DataFrame:
    """One row per article: popularity and recency signals."""
    last = train["week"].max()

    feats = (
        train.group_by("article_id").agg(
            pl.len().alias("art_total_buys"),
            pl.col("customer_id").n_unique().alias("art_unique_buyers"),
            pl.col("price").mean().alias("art_mean_price"),
            pl.col("week").max().alias("art_last_week"),
        )
        .with_columns(
            (last - pl.col("art_last_week")).alias("art_weeks_since_last_sold"),
        )
    )

    # Recent (last 4 weeks) popularity — a stronger signal than all-time.
    recent = (
        train.filter(pl.col("week") > last - 4)
        .group_by("article_id").agg(pl.len().alias("art_recent_buys"))
    )

    feats = feats.join(recent, on="article_id", how="left").with_columns(
        pl.col("art_recent_buys").fill_null(0)
    )
    return feats


def customer_features(train: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:
    """One row per customer: activity signals + age/channel from metadata."""
    last = train["week"].max()

    behav = (
        train.group_by("customer_id").agg(
            pl.len().alias("cust_total_buys"),
            pl.col("article_id").n_unique().alias("cust_unique_articles"),
            pl.col("price").mean().alias("cust_mean_price"),
            pl.col("week").max().alias("cust_last_week"),
            pl.col("sales_channel_id").mean().alias("cust_mean_channel"),
        )
        .with_columns(
            (last - pl.col("cust_last_week")).alias("cust_weeks_since_last_buy"),
        )
    )

    # Bring in age from the customers table.
    feats = behav.join(
        customers.select(["customer_id", "age"]).rename({"age": "cust_age"}),
        on="customer_id", how="left",
    )
    return feats


def interaction_features(candidates: pl.DataFrame, train: pl.DataFrame,
                         articles: pl.DataFrame) -> pl.DataFrame:
    """Per-pair signals: prior buys of this article, plus customer affinity
    to this article's department, product-type, and colour.

    Affinity features vary strongly across candidates (unlike the near-constant
    recency features), so they give the ranker real customer-article 'fit' signal.
    """
    # 1. Prior buys of this exact article.
    pair_counts = (
        train.group_by(["customer_id", "article_id"])
        .agg(pl.len().alias("cust_art_prior_buys"))
    )
    cands = candidates.join(pair_counts, on=["customer_id", "article_id"], how="left") \
                      .with_columns(pl.col("cust_art_prior_buys").fill_null(0))

    # Article attribute lookup.
    attrs = articles.select([
        "article_id", "product_type_name", "department_name", "colour_group_name",
    ])
    train_a = train.join(attrs, on="article_id", how="left")

    # 2. Build a customer-affinity feature for each attribute.
    for col, short in [
        ("product_type_name", "type"),
        ("department_name", "dept"),
        ("colour_group_name", "colour"),
    ]:
        aff = (
            train_a.group_by(["customer_id", col])
            .agg(pl.len().alias(f"cust_{short}_affinity"))
        )
        cands = (
            cands.join(attrs.select(["article_id", col]),
                       on="article_id", how="left")
            .join(aff, on=["customer_id", col], how="left")
            .with_columns(pl.col(f"cust_{short}_affinity").fill_null(0))
            .drop(col)
        )

    return cands


def assemble_training_data(candidates: pl.DataFrame, train: pl.DataFrame,
                           customers: pl.DataFrame, articles: pl.DataFrame,
                           valid: pl.DataFrame) -> pl.DataFrame:
    """Join all features onto candidates and attach the label.

    Label = 1 if the customer actually bought this article in the held-out
    week, else 0. This is the table the ranker trains on.
    """
    af = article_features(train)
    cf = customer_features(train, customers)

    data = (
        candidates
        .join(af, on="article_id", how="left")
        .join(cf, on="customer_id", how="left")
    )
    data = interaction_features(data, train, articles)

    # Label from ground truth: explode valid into (customer, article) positives.
    positives = (
        valid.explode("actual").rename({"actual": "article_id"})
        .with_columns(pl.lit(1).alias("label"))
    )
    data = data.join(positives, on=["customer_id", "article_id"], how="left") \
               .with_columns(pl.col("label").fill_null(0))
    return data


if __name__ == "__main__":
    from src.hnm.evaluation.split import load_transactions, train_valid_split

    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    customers = pl.read_parquet("data/processed/customers.parquet")

    af = article_features(train)
    cf = customer_features(train, customers)

    print(f"article features: {af.height:,} rows, {af.width} cols")
    print(af.head(3))
    print(f"\ncustomer features: {cf.height:,} rows, {cf.width} cols")
    print(cf.head(3))
