"""MAP@12 — the H&M competition metric.

For each customer:
    AP@12 = (1 / min(m, 12)) * sum_{k=1..12} P(k) * rel(k)
where m = number of items the customer actually bought, P(k) is precision
at cut-off k, and rel(k) is 1 if the k-th prediction is a true positive.
The score is the mean of AP@12 over all scored customers.

Key subtlety: the normalizer is min(m, 12), NOT m. A customer who bought
20 items can still score 1.0 with 12 correct predictions.
"""

from typing import Sequence


def apk(actual: Sequence[int], predicted: Sequence[int], k: int = 12) -> float:
    """Average precision at k for a single customer."""
    if not actual:
        return 0.0

    predicted = list(predicted)[:k]
    actual_set = set(actual)

    hits = 0
    score = 0.0
    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:  # count each hit once
            hits += 1
            # precision at this rank
            score += hits / (i + 1.0)

    return score / min(len(actual_set), k)


def mapk(actuals: Sequence[Sequence[int]], predictions: Sequence[Sequence[int]], k: int = 12) -> float:
    """Mean average precision at k over many customers."""
    return sum(apk(a, p, k) for a, p in zip(actuals, predictions)) / len(actuals)


if __name__ == "__main__":
    # Hand-worked test cases — verify the metric is correct before trusting it.

    # 1. Perfect single hit at rank 1, customer bought 1 item -> AP = 1.0
    assert abs(apk([10], [10, 20, 30]) - 1.0) < 1e-9

    # 2. Single relevant item, predicted at rank 2 -> P(2)=1/2, /min(1,12)=1 -> 0.5
    assert abs(apk([10], [20, 10, 30]) - 0.5) < 1e-9

    # 3. Two relevant items at ranks 1 and 2, m=2:
    #    hits: rank1 -> 1/1, rank2 -> 2/2 ; sum=2 ; /min(2,12)=2 -> 1.0
    assert abs(apk([10, 20], [10, 20, 30]) - 1.0) < 1e-9

    # 4. Two relevant, at ranks 1 and 3, m=2:
    #    rank1 -> 1/1=1.0 ; rank3 -> 2/3 ; sum=1.6667 ; /2 -> 0.8333
    assert abs(apk([10, 20], [10, 99, 20]) - (1.0 + 2/3) / 2) < 1e-9

    # 5. No hits -> 0.0
    assert abs(apk([10, 20], [1, 2, 3]) - 0.0) < 1e-9

    # 6. min(m,12) normalizer: bought 20 items, 12 correct predictions -> 1.0
    actual_20 = list(range(20))
    pred_12 = list(range(12))
    assert abs(apk(actual_20, pred_12) - 1.0) < 1e-9

    # 7. mapk averages two customers: 1.0 and 0.5 -> 0.75
    assert abs(mapk([[10], [10]], [[10], [20, 10]]) - 0.75) < 1e-9

    print("all MAP@12 tests passed.")
