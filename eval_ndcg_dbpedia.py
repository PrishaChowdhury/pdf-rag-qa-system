"""
NDCG evaluation harness using REAL data from the DBpedia-Entity v2 test collection
(Hasibi et al., "DBpedia-Entity v2: A Test Collection for Entity Search", SIGIR 2017).
https://github.com/iai-group/DBpedia-Entity

Unlike evandcg.py (synthetic, easy-to-separate topics), this uses one query from the
REAL benchmark -- "roman architecture" (query id INEX_LD-2009039) -- along with its
ACTUAL, published relevance judgments (0 = irrelevant, 1 = relevant, 2 = highly relevant),
pulled directly from qrels-v2.txt. Because every candidate entity is from the SAME domain
(architecture/history), this is a genuinely harder retrieval task than eval_ndcg.py: weak
embeddings are much more likely to rank a grade-0 distractor (e.g. Almudena Cathedral, a
completely different cathedral) above a grade-2 answer (e.g. Roman temple).

Entity descriptions below are original summaries written from general knowledge of each
(real, well-documented) topic -- not scraped verbatim from Wikipedia/DBpedia -- used here
purely as retrievable text for the embedding comparison. The query, entity selection, and
relevance grades are the authentic benchmark data.

Run with:  python eval_ndcg_dbpedia.py
Requires:  GOOGLE_API_KEY set in .env (same as app.py) for the Gemini embedding strategy.
"""

import os
import math
import csv
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

QUERY = "roman architecture"  # DBpedia-Entity v2 query INEX_LD-2009039

# Real entities from the query's judgment pool + their real relevance grades (qrels-v2.txt)
ENTITIES = {
    # --- Grade 2: highly relevant (the correct answers) ---
    "Ancient_Roman_architecture": (
        "Ancient Roman architecture refers to the building style developed by the Romans, "
        "known for its use of concrete, the arch, and the dome. It produced structures such "
        "as temples, basilicas, amphitheatres, and aqueducts across the empire.", 2),
    "Roman_temple": (
        "A Roman temple was a structure built to house a deity within a Roman city, typically "
        "featuring a raised platform, a columned porch, and a rectangular cella where the cult "
        "statue stood, drawing heavily on earlier Etruscan and Greek temple design.", 2),
    "Triumphal_arch": (
        "A triumphal arch is a free-standing monumental structure with one or more arched "
        "passageways, built by the Romans to commemorate military victories or important "
        "events, often decorated with relief sculpture depicting the honored achievement.", 2),
    "Roman_bridge": (
        "Roman bridges were among the first large, permanent bridges built, using the arch and "
        "Roman concrete to span rivers and valleys. Many, built for military roads and aqueducts, "
        "remain standing and in some cases still in use today.", 2),
    "Basilica": (
        "In Roman architecture, a basilica was a large public building used for law courts, "
        "administration, and commerce, characterized by a rectangular hall with aisles divided "
        "by columns; the form was later adapted for early Christian churches.", 2),
    "Peristyle": (
        "A peristyle is a continuous porch formed by a row of columns surrounding a building, "
        "courtyard, or garden, widely used in Roman villas and temples to frame open, colonnaded "
        "spaces.", 2),
    "Coffer": (
        "A coffer is a sunken panel in the shape of a square, rectangle, or octagon in a ceiling, "
        "soffit, or vault, used extensively by Roman architects (most famously in the Pantheon's "
        "dome) to reduce weight while adding decorative structure.", 2),
    "Insula_(building)": (
        "An insula was a Roman apartment building that housed most of the urban population, "
        "typically several stories tall with shops on the ground floor and residential units "
        "above, representing an early form of high-density urban housing.", 2),

    # --- Grade 1: related but not a direct answer ---
    "Colosseum": (
        "The Colosseum is a large amphitheatre in the centre of Rome, built under the Flavian "
        "emperors, used for gladiatorial contests and public spectacles, and is the largest "
        "ancient amphitheatre ever constructed.", 1),
    "Roman_aqueduct": (
        "A Roman aqueduct was an engineered channel built to carry water from a distant source "
        "into cities and towns, relying on gravity and a slight, continuous downward gradient, "
        "often carried across valleys on arched bridges.", 1),
    "Pantheon,_Rome": (
        "The Pantheon is a former Roman temple in Rome, famous for its massive coffered concrete "
        "dome with a central oculus, which remained the largest dome in the world for over a "
        "thousand years.", 1),
    "Roman_Empire": (
        "The Roman Empire was the post-Republican period of ancient Rome, spanning centuries of "
        "territorial expansion, governance, and cultural influence across Europe, North Africa, "
        "and the Middle East.", 1),
    "Vitruvius": (
        "Vitruvius was a Roman author, architect, and engineer whose treatise De Architectura is "
        "the only major work on architecture to survive from antiquity, influencing architectural "
        "theory for centuries.", 1),
    "Roman_concrete": (
        "Roman concrete was a building material made from a mix of volcanic ash, lime, and "
        "aggregate, which allowed Romans to construct large vaulted and domed structures and "
        "proved remarkably durable, with some structures surviving to this day.", 1),
    "Classical_architecture": (
        "Classical architecture refers to the architectural styles of ancient Greece and Rome, "
        "characterized by symmetry, proportion, and orders such as Doric, Ionic, and Corinthian "
        "columns, which influenced later Renaissance and Neoclassical design.", 1),
    "Villa": (
        "A villa was originally a Roman upper-class country house, often built as a working "
        "agricultural estate or a leisure retreat, later becoming a broader term for detached, "
        "usually luxurious residences.", 1),

    # --- Grade 0: irrelevant distractors (same broad domain, wrong answer) ---
    "Almudena_Cathedral": (
        "Almudena Cathedral is a Catholic cathedral in Madrid, Spain, completed in the late 20th "
        "century, blending neoclassical, neo-gothic, and neo-romanesque architectural styles.", 0),
    "Baroque_architecture": (
        "Baroque architecture is an ornate, dramatic style that originated in late 16th-century "
        "Italy, characterized by grandeur, curved forms, and elaborate decoration, later spreading "
        "across Catholic Europe and Latin America.", 0),
    "Islamic_architecture": (
        "Islamic architecture encompasses building styles developed across the Muslim world, "
        "identifiable by features such as domes, minarets, arches, and geometric or calligraphic "
        "ornamentation, seen in mosques, palaces, and tombs.", 0),
    "Medieval_architecture": (
        "Medieval architecture spans styles from the fall of Rome through the late 15th century, "
        "including Romanesque and Gothic architecture, seen prominently in European cathedrals "
        "and castles.", 0),
    "Architecture_of_Germany": (
        "The architecture of Germany reflects a wide range of historical styles, from Romanesque "
        "and Gothic cathedrals to Bauhaus modernism, shaped by the country's regional and "
        "political history.", 0),
    "Granada_Cathedral": (
        "Granada Cathedral is a Roman Catholic cathedral in Granada, Spain, built primarily in "
        "the Spanish Renaissance style beginning in the 16th century on the site of a former "
        "mosque.", 0),
    "Church_architecture": (
        "Church architecture is the design and construction of Christian places of worship, "
        "encompassing a wide variety of historical styles from early basilicas through Gothic, "
        "Baroque, and modern forms.", 0),
    "Ostrogothic_Kingdom": (
        "The Ostrogothic Kingdom was a state established in Italy and neighboring areas by the "
        "Ostrogoths at the end of the 5th century, following the fall of the Western Roman "
        "Empire.", 0),
}

RELEVANCE_MAP = {name: grade for name, (_, grade) in ENTITIES.items()}
CHUNKS = {name: text for name, (text, _) in ENTITIES.items()}

K_VALUES = [5, 10]


def dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranked_names, relevance_map, k):
    top_k = ranked_names[:k]
    gains = [relevance_map.get(name, 0) for name in top_k]
    ideal_gains = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_gains += [0] * (k - len(ideal_gains))
    idcg = dcg(ideal_gains)
    if idcg == 0:
        return 0.0
    return dcg(gains) / idcg


def build_retriever(embeddings):
    names = list(CHUNKS.keys())
    texts = [CHUNKS[n] for n in names]
    return FAISS.from_texts(texts, embeddings, metadatas=[{"name": n} for n in names])


def evaluate_strategy(embeddings, label):
    vectorstore = build_retriever(embeddings)
    results = vectorstore.similarity_search(QUERY, k=len(CHUNKS))
    ranked_names = [r.metadata["name"] for r in results]
    scores = {k: ndcg_at_k(ranked_names, RELEVANCE_MAP, k) for k in K_VALUES}
    return ranked_names, scores


def main():
    strategies = [
        ("all-MiniLM-L6-v2",
         HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})),
        ("all-mpnet-base-v2",
         HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})),
    ]

    google_key = os.getenv("GOOGLE_API_KEY", "")
    if google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        strategies.append((
            "Google gemini-embedding-001",
            GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_key),
        ))
    else:
        print("GOOGLE_API_KEY not found -- skipping Google embedding strategy.")

    print(f"Query: '{QUERY}'  ({len(CHUNKS)} candidate entities, real DBpedia-Entity v2 judgments)\n")

    all_results = {}
    for label, embeddings in strategies:
        ranked_names, scores = evaluate_strategy(embeddings, label)
        all_results[label] = (ranked_names, scores)
        score_str = "  ".join(f"NDCG@{k}={v:.4f}" for k, v in scores.items())
        print(f"{label:28s}  {score_str}")
        print(f"  Top 5 retrieved: {ranked_names[:5]}")
        print()

    with open("ndcg_dbpedia_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy"] + [f"NDCG@{k}" for k in K_VALUES] + ["top_5_retrieved"])
        for label, (ranked_names, scores) in all_results.items():
            writer.writerow([label] + [f"{scores[k]:.4f}" for k in K_VALUES] + [", ".join(ranked_names[:5])])

    print("Saved results to ndcg_dbpedia_results.csv")


if __name__ == "__main__":
    main()