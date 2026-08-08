"""Run candidate generators on the real split and report recall."""
import polars as pl

from src.hnm.evaluation.split import load_transactions, train_valid_split
from src.hnm.retrieval.candidates import repurchase_candidates, popularity_candidates, candidate_recall


def main():
    tx = load_transactions()
    train, valid, wk = train_valid_split(tx)
    print(f"holdout week: {wk}, valid customers: {valid.height:,}\n")

    valid_customers = valid["customer_id"]

    # Repurchase at the chosen 52-week window.
    repurchase = repurchase_candidates(train, recent_weeks=52)
    r_stats = candidate_recall(repurchase, valid)
    print(f"repurchase (52wk):        recall={r_stats['recall']:.4f}  "
          f"avg_cands={r_stats['avg_candidates_per_customer']:.1f}")

    # Popularity at several pool sizes.
    for n in [50, 100, 200, 500]:
        pop = popularity_candidates(
            train, valid_customers, n=n, recent_weeks=2)
        p_stats = candidate_recall(pop, valid)

        # Combined = repurchase + popularity, deduped.
        combined = pl.concat([
            repurchase.select(["customer_id", "article_id", "method"]),
            pop.select(["customer_id", "article_id", "method"]),
        ])
        c_stats = candidate_recall(combined, valid)

        print(f"popular top-{n:<3}:          recall={p_stats['recall']:.4f}  "
              f"| combined recall={c_stats['recall']:.4f}  "
              f"avg_cands={c_stats['avg_candidates_per_customer']:.1f}")


if __name__ == "__main__":
    main()
