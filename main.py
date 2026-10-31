"""Week 03 entry point: runs every section in checklist order.

Usage:  python main.py | tee output/run_log.txt
"""
import os
import traceback

from classification import run_text_classification
from preprocessing import run_preprocessing_demo
from tokenization import run_tokenization_demo
from embeddings import run_embeddings_demo
from semantic_search import run_semantic_search

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_section(title, func):
    """Run one section; report any exception without stopping the other sections."""
    banner(title)
    try:
        func()
        print(f"\n[OK] {title}")
    except Exception:
        print(f"\n[FAILED] {title}")
        traceback.print_exc()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_section("1. Machine Learning - Text Classification", run_text_classification)
    run_section("2. NLP - Text Preprocessing", run_preprocessing_demo)
    run_section("3. Tokenization", run_tokenization_demo)
    run_section("4. Embeddings and Semantic Similarity", run_embeddings_demo)
    run_section("5. Semantic Search", lambda: run_semantic_search(OUTPUT_DIR))
    banner("Finished")


if __name__ == "__main__":
    main()
