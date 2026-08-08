"""Candidate generation — the retrieval stage of the funnel.

Each generator returns a Polars DataFrame with columns:
    customer_id, article_id, method
'method' tags where the candidate came from, so we can measure each
strategy's contribution and later feed it as a feature to the ranker.

We evaluate candidates by RECALL against the held-out week: what fraction
of true purchases appear somewhere in the candidate set. Recall is the
ceiling on final MAP@12 — the ranker can only reorder what retrieval finds.
"""

import polars as pl


def repurchase_candidates(train: pl.DataFrame, recent_weeks: int = 12) -> pl.DataFrame:
    """Each customer's own articles bought in the last `recent_weeks` weeks.

    Rationale (from EDA): ~14% of purchases repeat a prior article, and those
    repeats are highly predictable. Cheap, high-precision candidates.
    """
    last = train["week"].max()
    pool = train.filter(pl.col("week") > last - recent_weeks)
    cands = (
        pool.select(["customer_id", "article_id"])
        .unique()
        .with_columns(pl.lit("repurchase").alias("method"))
    )
    return cands


def candidate_recall(candidates: pl.DataFrame, valid: pl.DataFrame) -> dict:
    """What fraction of true (customer, article) purchases are in candidates?

    Recall = (true purchases that appear as a candidate) / (all true purchases),
    over customers who bought something in the held-out week.
    """
    # One row per (customer, true article).
    truth = valid.explode("actual").rename({"actual": "article_id"})

    # Distinct candidate pairs, tagged so we can detect matches after the join.
    cand_pairs = (
        candidates.select(["customer_id", "article_id"])
        .unique()
        .with_columns(pl.lit(1).alias("is_cand"))
    )

    covered = truth.join(
        cand_pairs, on=["customer_id", "article_id"], how="left")
    hit_truths = covered["is_cand"].fill_null(0).sum()
    total_truths = truth.height

    per_cust = candidates.group_by("customer_id").len()["len"].mean()

    return {
        "recall": hit_truths / total_truths,
        "true_purchases": total_truths,
        "covered": int(hit_truths),
        "avg_candidates_per_customer": per_cust,
    }


def popularity_candidates(train: pl.DataFrame, valid_customers: pl.Series,
                          n: int = 100, recent_weeks: int = 2) -> pl.DataFrame:
    """Top-n recently popular articles, proposed to every scored customer.

    Rationale (from EDA): fashion is long-tailed, so a broad recent-popularity
    pool is the main recall source, especially for cold-start customers with
    no purchase history of their own.
    """
    last = train["week"].max()
    pool = train.filter(pl.col("week") > last - recent_weeks)
    top = (
        pool.group_by("article_id").len()
        .sort("len", descending=True)
        .head(n)["article_id"]
    )
    # Cross-join: every scored customer gets every popular article.
    cands = (
        pl.DataFrame({"customer_id": valid_customers})
        .join(pl.DataFrame({"article_id": top}), how="cross")
        .with_columns(pl.lit("popular").alias("method"))
    )
    return cands
