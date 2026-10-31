"""Section 3: Tokenization - subwords, token IDs, special tokens."""
import re

SENTENCE = "Unbelievably, the tokenizers don't agree on this sentence!"
CORPUS = [SENTENCE, "Tokenization splits text into smaller units.", "Hello world, hello tokens."]


class SimpleWordTokenizer:
    """Offline stand-in for BERT: word/punctuation tokens with a few special tokens."""

    def __init__(self, corpus):
        vocab_words = {tok for s in corpus for tok in self.tokenize(s)}
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for word in sorted(vocab_words):
            self.vocab.setdefault(word, len(self.vocab))
        self.inverse = {i: t for t, i in self.vocab.items()}
        self.special_tokens_map = {"unk_token": "[UNK]", "cls_token": "[CLS]",
                                   "sep_token": "[SEP]", "pad_token": "[PAD]"}

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]

    def decode(self, ids):
        return " ".join(self.inverse.get(i, "[UNK]") for i in ids)


class CharacterTokenizer:
    """Offline stand-in for a second tokenizer: one token per character."""

    def tokenize(self, text):
        return list(text)


def run_tokenization_demo(sentence=SENTENCE):
    try:
        from transformers import AutoTokenizer
        bert = AutoTokenizer.from_pretrained("bert-base-uncased")
        second = AutoTokenizer.from_pretrained("gpt2")
        second_name = "gpt2"
        mode = "transformers (bert-base-uncased, gpt2)"
    except Exception as exc:
        bert = SimpleWordTokenizer(CORPUS)
        second = CharacterTokenizer()
        second_name = "character-level"
        mode = f"offline fallback ({type(exc).__name__}: pretrained tokenizers unavailable)"
    print(f"Mode: {mode}\n")

    print(f"Sentence: {sentence}")
    tokens = bert.tokenize(sentence)
    print(f"1. Tokens:    {tokens}")

    ids = bert.convert_tokens_to_ids(tokens)
    print(f"2. Token IDs: {ids}")

    decoded = bert.decode(ids)
    print(f"3. Decoded:   {decoded}\n")

    print("Special tokens:", bert.special_tokens_map)

    print("\nToken counts for two tokenizers:")
    first_count = len(tokens)
    second_count = len(second.tokenize(sentence))
    print(f"  first  (bert/word) : {first_count} tokens")
    print(f"  second ({second_name}) : {second_count} tokens")

    print("\nSubword tokenization: rare words are split into smaller known pieces")
    print("(BERT marks continuation pieces with '##'), so unseen words can still be represented.")
    print("Special tokens such as [CLS] (start/classification) and [SEP] (separator) mark")
    print("structure for the model; [PAD] fills batches to equal length and [UNK] marks unknowns.")
    return bert
