"""Section 4: Embeddings and semantic similarity."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Returns dense vectors for texts.

    Uses a sentence-transformer model when it can be loaded. Otherwise it falls
    back to TF-IDF vectors, which measure shared words rather than meaning, and
    the `backend` attribute says which one is active.
    """

    def __init__(self, fallback_corpus=None):
        self.model = None
        self.vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(SENTENCE_MODEL_NAME)
            self.backend = f"sentence-transformers: {SENTENCE_MODEL_NAME}"
        except Exception as exc:
            self.backend = f"TF-IDF fallback ({type(exc).__name__}: sentence model unavailable)"
            self.vectorizer = TfidfVectorizer()
            if fallback_corpus is not None:
                self.vectorizer.fit(fallback_corpus)

    def encode(self, texts):
        texts = list(texts)
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        if not hasattr(self.vectorizer, "vocabulary_"):
            self.vectorizer.fit(texts)
        return normalize(self.vectorizer.transform(texts).toarray())


def cosine(vec_a, vec_b):
    return float(cosine_similarity(np.reshape(vec_a, (1, -1)), np.reshape(vec_b, (1, -1)))[0, 0])


def interpret(score):
    if score >= 0.6:
        return "strongly similar"
    if score >= 0.3:
        return "somewhat related"
    return "unrelated"


PAIRS = [
    ("The cat sat on the mat.", "A kitten is resting on a rug.", "semantically similar"),
    ("The cat sat on the mat.", "Stock prices fell sharply today.", "unrelated"),
    ("I love reading books at night.", "Reading novels before bed is my favourite hobby.", "semantically similar"),
    ("I love reading books at night.", "The train to the airport leaves at noon.", "unrelated"),
]


def run_embeddings_demo():
    all_sentences = [s for pair in PAIRS for s in pair[:2]]
    embedder = Embedder(fallback_corpus=all_sentences)
    print(f"Embedding backend: {embedder.backend}\n")

    print("Pairwise cosine similarity (embeddings):")
    for first, second, expected in PAIRS:
        vec_a, vec_b = embedder.encode([first, second])
        score = cosine(vec_a, vec_b)
        print(f"  [{expected}] score={score:.3f} ({interpret(score)})")
        print(f"     A: {first}\n     B: {second}")

    print("\nComparison with word counts (CountVectorizer, no meaning captured):")
    counter = CountVectorizer().fit(all_sentences)
    for first, second, expected in PAIRS:
        counts = counter.transform([first, second]).toarray()
        score = cosine(counts[0], counts[1])
        print(f"  [{expected}] word-count cosine={score:.3f}")

    print("\nInterpretation:")
    print("- Scores near 1 mean the vectors point the same way; near 0 means little in common.")
    print("- Word counts only reward exact shared words, so 'cat/mat' and 'kitten/rug' score low.")
    print("- Sentence embeddings place sentences with similar meaning close together even with")
    print("  different words. The TF-IDF fallback cannot do this, so run with sentence-transformers")
    print("  installed (see README) to see the full semantic behaviour.")
    return embedder
