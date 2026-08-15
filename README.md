# H&M Personalized Fashion Recommendations

A two-stage recommender (retrieval, then ranking) built from scratch for the
[H&M Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
The job: recommend 12 articles to each customer for the coming week, out of a catalogue of
105K articles, across 1.37M customers and 31.8M past transactions. Scored with MAP@12.

I built this to be the same shape as the competition's strong solutions - narrow the
catalogue down with cheap retrieval, then rank the survivors with a learning-to-rank model -
and to run the whole thing on a single 16GB laptop with Polars and LightGBM.

## Results

| Stage | MAP@12 | |
|-------|--------|--|
| Popularity baseline | 0.00900 | same 12 bestsellers for everyone |
| Two-stage ranker (repurchase + popularity) | 0.04265 | held-out buyers |
| + item2item candidates | 0.04365 | held-out buyers |
| + affinity features (dept / type / colour) | **0.04473** | held-out buyers, roughly 5x the baseline |


## Why two stages

Scoring every customer against every article would be 1.37M x 105K, about 144 billion pairs.
That is not going to happen on a laptop, so the pipeline works as a funnel:

- **Retrieval** proposes a few dozen plausible articles per customer. It only needs to be
  cheap and to catch the real purchases somewhere in its net. This stage sets the ceiling -
  the ranker can only reorder what retrieval hands it.
- **Ranking** takes those candidates and orders them with a LightGBM ranker, keeping the top
  12. It optimises the order within each customer, which is exactly what MAP@12 cares about.

```
105K articles  ->  retrieval  ->  a few dozen candidates  ->  LightGBM ranker  ->  top 12
```

## The candidate strategies

Three of them, each measured by recall against the held-out week:

| Strategy | What it proposes | Recall alone |
|----------|------------------|--------------|
| Repurchase | the customer's own recent purchases | 0.037 |
| Popularity | recent bestsellers, to everyone | 0.105 |
| item2item | "people who bought X also bought Y" | 0.046 |

Combined recall is 0.159. They stack because they catch different things - repurchase is
precise but narrow, popularity is the broad net (and the fallback for customers with no
history), and item2item picks up substitutes and pairings the other two miss. The EDA made
this concrete: only about 14% of purchases are repeats, so most of the recall has to come
from popularity rather than a customer's own history.

## The features

Each candidate gets 15 features before the ranker sees it:

- **Article** - how much and how recently it sells, unique buyers, average price.
- **Customer** - how often they buy, how many distinct items, average price, main channel,
  how recently they last bought, age.
- **Interaction** - prior purchases of this exact article, plus how much this customer tends
  to buy this article's product type, department, and colour.

Those last three (the affinity features) came out of looking at feature importances and
noticing the ranker had almost nothing describing customer-to-article fit. Adding them moved
the score from 0.0437 to 0.0447 - a small but real gain from a diagnosis rather than a guess.

## Some engineering notes

- Everything runs on Polars. The raw 3.3GB transactions CSV becomes a 208MB parquet file
  (about 16x smaller) that loads in seconds. The 64-character hex customer IDs get parsed
  down to UInt64, article IDs to Int32.
- The full-population inference runs in batches of 200K customers so memory stays bounded.
  This started as a problem - building candidates for all 1.37M at once blew past 16GB - and
  batching was the fix, which is a more honest reflection of real-world constraints anyway.
- The MAP@12 implementation has seven unit tests, including the awkward min(m, 12)
  normalisation that is easy to get wrong.
- Validation is a time-based split: hold out the final week, train on everything before it.
  That mirrors the actual task of predicting the next seven days.

## Layout

```
src/hnm/
  data/reduce_memory.py      CSV -> compact parquet
  retrieval/candidates.py    repurchase, popularity, item2item
  features/build.py          article / customer / interaction features
  ranking/train.py           LightGBM ranker
  ranking/infer.py           batched full-population inference
  evaluation/split.py        time-based split
  evaluation/metric.py       MAP@12 (with tests)
  evaluation/evaluate.py     popularity baseline
scripts/                     eval_candidates, build_features, score_ranker
notebooks/02_eda.ipynb       exploratory analysis
```

## Running it

```bash
python -m venv .venv && source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt

# download the 4 competition CSVs into data/raw/ first (Kaggle API), then:
python -m src.hnm.data.reduce_memory      # CSV -> parquet
python -m src.hnm.evaluation.evaluate     # popularity baseline
python -m scripts.eval_candidates         # candidate recall
python -m scripts.build_features          # build the training table
python -m src.hnm.ranking.train           # train the ranker
python -m scripts.score_ranker            # held-out-buyers MAP@12
python -m src.hnm.ranking.infer           # true all-customer MAP@12
```

The data, model, and feature files are gitignored since they are large and fully
regenerable from the commands above.

## Stack

Python, Polars, LightGBM, scikit-learn, PyArrow, NumPy, pandas, Jupyter.