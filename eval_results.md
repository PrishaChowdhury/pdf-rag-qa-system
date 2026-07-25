# Embedding Strategy Evaluation

Benchmarked against a query from the [DBpedia-Entity v2](https://github.com/iai-group/DBpedia-Entity) test collection ("roman architecture"), using its real, published relevance judgments (0 = irrelevant, 1 = relevant, 2 = highly relevant) as ground truth. See `eval_ndcg_dbpedia.py`.

| Model | NDCG@3 |
|---|---|
| all-MiniLM-L6-v2 | 0.8827 |
| all-mpnet-base-v2 | 0.7346 |
| Google gemini-embedding-001 | 1.0000 |

**Gemini's embedding model performed best**, retrieving only highly-relevant (grade 2) results in its top 3, because it captured the query's intent more precisely than the two Hugging Face models — both of which ranked a topically-related-but-incorrect result (e.g. `Vitruvius`, a person, not an architectural style/structure) above a genuinely correct one, a mistake NDCG is specifically designed to penalize.
