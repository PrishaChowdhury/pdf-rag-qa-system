"""
NDCG evaluation harness for comparing embedding strategies in the PDF RAG QA System.

What this does:
  1. Defines a small fixed set of "documents" (the same 7 sections in test_document.pdf)
  2. Defines a set of test questions, each with a hand-labeled relevance grade
     (2 = directly answers the question, 1 = related but not the answer, 0 = irrelevant)
  3. Embeds the documents with each of 3 embedding strategies, builds a FAISS index per
     strategy, retrieves the top-k chunks for each question, and scores the ranking with NDCG@k
  4. Prints a comparison table and saves results to ndcg_results.csv

Run with:  python eval_ndcg.py
Requires:  GOOGLE_API_KEY set in .env (same as app.py) for the Gemini embedding strategy.
"""

import os
import math
import csv
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# 1. Ground-truth documents (mirrors the sections in test_document.pdf)
# ---------------------------------------------------------------------------
CHUNKS = {
    0: "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
       "light energy into chemical energy stored in glucose, using chlorophyll in chloroplasts. "
       "The light-dependent reactions produce ATP and NADPH; the Calvin cycle uses them to turn "
       "carbon dioxide into glucose, releasing oxygen as a byproduct.",
    1: "The French Revolution began in 1789, driven by frustration with the absolute monarchy of "
       "Louis XVI, economic hardship, and Enlightenment ideas. The storming of the Bastille on "
       "July 14, 1789 began the revolution. The Reign of Terror, led by Robespierre's Committee "
       "of Public Safety, executed thousands before Napoleon Bonaparte seized power in 1799.",
    2: "Machine learning is a subfield of AI focused on learning patterns from data. Supervised "
       "learning trains on labeled examples for classification and regression; unsupervised "
       "learning finds structure in unlabeled data via clustering and dimensionality reduction. "
       "Neural networks and deep learning drive advances in vision, NLP, and generative AI.",
    3: "The water cycle describes water's continuous movement through evaporation, transpiration, "
       "condensation, precipitation, runoff, and infiltration. Sunlight evaporates surface water "
       "into vapor, which condenses into clouds and falls as rain, snow, sleet, or hail, "
       "eventually returning to rivers, oceans, and groundwater aquifers.",
    4: "Macronutrients -- carbohydrates, proteins, and fats -- are needed in large amounts for "
       "energy and body function. Carbohydrates break down into glucose for fuel. Proteins, made "
       "of amino acids, build and repair tissue and muscle. Fats provide concentrated energy and "
       "support absorption of vitamins A, D, E, and K.",
    5: "Renewable energy includes solar power (photovoltaic cells converting sunlight to "
       "electricity), wind power (turbines converting moving air to electricity), hydroelectric "
       "power (flowing water spinning turbines), and geothermal energy (heat from beneath the "
       "Earth's surface). These reduce reliance on fossil fuels and cut greenhouse emissions.",
    6: "Ancient Roman architecture is known for concrete construction, the true arch, and the "
       "dome, seen in the Pantheon and Colosseum. Roman aqueducts used arches and gravity to "
       "transport water over long distances to cities. Roman roads used layered, durable "
       "materials to connect the empire for trade and military movement.",
}

# ---------------------------------------------------------------------------
# 2. Test queries with graded relevance judgments {chunk_id: relevance_grade}
#    Only chunks listed are relevant; anything omitted is treated as grade 0.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    ("How do plants turn sunlight into energy?", {0: 2, 5: 1}),
    ("What event marked the start of the uprising against the French king?", {1: 2}),
    ("What's the difference between supervised and unsupervised learning?", {2: 2}),
    ("How does rain form from evaporated water?", {3: 2, 0: 1}),
    ("Which nutrient should I eat more of to repair muscle?", {4: 2}),
    ("What are some clean alternatives to burning coal for electricity?", {5: 2}),
    ("How did the Romans manage to transport water over long distances?", {6: 2, 3: 1}),
    ("What gives the most notorious period of the French Revolution its name?", {1: 2}),
    ("How is glucose used as fuel in the body?", {4: 2, 0: 1}),
    ("What material let the Romans build durable domes and arches?", {6: 2}),
]

K = 3  # evaluate top-3 retrieved chunks


def dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranked_chunk_ids, relevance_map, k=K):
    top_k = ranked_chunk_ids[:k]
    gains = [relevance_map.get(cid, 0) for cid in top_k]
    ideal_gains = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_gains += [0] * (k - len(ideal_gains))
    idcg = dcg(ideal_gains)
    if idcg == 0:
        return 0.0
    return dcg(gains) / idcg


def build_retriever(embeddings):
    ids = list(CHUNKS.keys())
    texts = [CHUNKS[i] for i in ids]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=[{"chunk_id": i} for i in ids])
    return vectorstore


def evaluate_strategy(name, embeddings):
    print(f"\nEmbedding: {embeddings.model if hasattr(embeddings, 'model') else name}")
    vectorstore = build_retriever(embeddings)
    scores = []
    for query, relevance_map in TEST_QUERIES:
        results = vectorstore.similarity_search(query, k=K)
        ranked_ids = [r.metadata["chunk_id"] for r in results]
        score = ndcg_at_k(ranked_ids, relevance_map, k=K)
        scores.append(score)
    avg = sum(scores) / len(scores)
    return scores, avg


def main():
    strategies = []

    # Strategy 1: current model used in app.py
    strategies.append((
        "all-MiniLM-L6-v2",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}),
    ))

    # Strategy 2: larger, higher-quality HF model
    strategies.append((
        "all-mpnet-base-v2",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}),
    ))

    # Strategy 3: Google embedding model (requires GOOGLE_API_KEY in .env)
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        strategies.append((
            "Google gemini-embedding-001",
            GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_key),
        ))
    else:
        print("GOOGLE_API_KEY not found in .env -- skipping Google embedding strategy.")

    all_results = {}
    for name, embeddings in strategies:
        scores, avg = evaluate_strategy(name, embeddings)
        all_results[name] = (scores, avg)

    print("\n" + "=" * 60)
    print(f"NDCG@{K} RESULTS")
    print("=" * 60)
    for name, (scores, avg) in all_results.items():
        print(f"{name:28s}  avg NDCG@{K} = {avg:.4f}")

    # Save per-query breakdown to CSV
    with open("ndcg_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["query"] + list(all_results.keys())
        writer.writerow(header)
        for i, (query, _) in enumerate(TEST_QUERIES):
            row = [query] + [f"{all_results[name][0][i]:.4f}" for name in all_results]
            writer.writerow(row)
        writer.writerow(["AVERAGE"] + [f"{all_results[name][1]:.4f}" for name in all_results])

    print("\nSaved per-query breakdown to ndcg_results.csv")


if __name__ == "__main__":
    main()