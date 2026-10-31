"""Section 5: Semantic search over a small document collection."""
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import Embedder

DOCUMENTS = [
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "Boil fresh pasta in salted water for about ten minutes, then drain and serve with sauce.",
    "The stock market closed higher today as technology shares rallied.",
    "Regular exercise improves cardiovascular health and strengthens muscles.",
    "Python is a popular programming language known for its readable syntax.",
    "The Roman Empire built roads, aqueducts, and fortifications across Europe.",
    "Vaccines train the immune system to recognise and fight specific pathogens.",
    "To make a simple tomato sauce, saute garlic and onions, then add crushed tomatoes.",
    "Neural networks learn patterns from data by adjusting their weights during training.",
    "Hiking trails in the mountains offer scenic views and fresh air.",
]

QUERIES = [
    "how do I cook pasta",
    "how do plants make food from light",
    "machine learning model training",
    "healthy heart exercise",
]


def semantic_search(query, embedder, doc_embeddings, top_k=3):
    """Return the top_k (index, score) pairs ranked by cosine similarity."""
    query_vec = embedder.encode([query])
    scores = cosine_similarity(query_vec, doc_embeddings)[0]
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked]


def run_semantic_search(output_dir):
    embedder = Embedder(fallback_corpus=DOCUMENTS + QUERIES)
    print(f"Embedding backend: {embedder.backend}")

    # Generate and store document embeddings once, then reuse them for every query.
    doc_embeddings = embedder.encode(DOCUMENTS)
    store_path = os.path.join(output_dir, "document_embeddings.npy")
    np.save(store_path, doc_embeddings)
    print(f"Stored {len(DOCUMENTS)} document embeddings with shape {doc_embeddings.shape} -> {store_path}\n")

    for query in QUERIES:
        print(f"Query: '{query}'")
        for rank, (idx, score) in enumerate(semantic_search(query, embedder, doc_embeddings), start=1):
            print(f"  {rank}. ({score:.3f}) {DOCUMENTS[idx]}")
        print()
    return doc_embeddings
