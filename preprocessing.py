"""Section 2: Natural Language Processing - text preprocessing."""
import re
import string


def clean_text(text, lowercase=True, remove_punctuation=True):
    """Return a normalised copy of `text`.

    Steps: optional lowercasing, optional punctuation removal, then collapsing
    runs of whitespace (spaces, tabs, newlines) into single spaces and trimming
    the ends.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text).__name__}")
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


SAMPLE_SENTENCES = [
    "  Hello,    WORLD!!!  ",
    "NLP\tis   fun...\nWe're learning   it today.",
    "Email me at test@example.com; it's #1 priority!",
    "   already clean sentence   ",
    "Numbers like 42 and 3.14 stay, punctuation goes (mostly).",
]


def run_preprocessing_demo():
    # Simple assertions act as tests for the core behaviour.
    assert clean_text("  Hello,    WORLD!!!  ") == "hello world"
    assert clean_text("A\n\nB\tC") == "a b c"
    assert clean_text("Keep CASE!", lowercase=False) == "Keep CASE"
    assert clean_text("keep, punctuation.", remove_punctuation=False) == "keep, punctuation."
    print("Unit checks passed.\n")

    for sentence in SAMPLE_SENTENCES:
        print(f"raw   : {sentence!r}")
        print(f"clean : {clean_text(sentence)!r}\n")

    print("Why preprocessing helps:")
    print("- Lowercasing stops 'Good' and 'good' being treated as different features.")
    print("- Removing punctuation and extra whitespace reduces noise in the vocabulary.")
    print("- A consistent form makes counts, matching and models behave more predictably.")
    print("- Some tasks need the opposite (e.g. sentiment with '!!!' or named entities in case),")
    print("  so each step is optional and chosen per task.")
