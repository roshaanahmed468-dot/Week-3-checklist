"""Section 1: Machine learning - text classification with scikit-learn."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

# Hand-written sentiment dataset: (text, label). Sentiment words are shared across
# examples so the model can learn them from 20 positive and 20 negative sentences.
DATASET = [
    ("I loved this movie, the acting was great", "positive"),
    ("What a wonderful experience, the staff were great", "positive"),
    ("The food was delicious and the service was great", "positive"),
    ("This book is fantastic, I loved every chapter", "positive"),
    ("Amazing view, the hotel room was clean and lovely", "positive"),
    ("The course was engaging and the teacher was great", "positive"),
    ("Absolutely delightful show, I was very happy", "positive"),
    ("The app works perfectly and is easy to use", "positive"),
    ("I had a wonderful time with friends at the park", "positive"),
    ("The new phone is fast, beautiful, and reliable", "positive"),
    ("Excellent quality product, I loved it", "positive"),
    ("The concert was joyful and the music was brilliant", "positive"),
    ("The room was lovely and the breakfast was excellent", "positive"),
    ("I enjoyed the trip, the guide was friendly and great", "positive"),
    ("The dinner was tasty and the waiter was lovely", "positive"),
    ("Great movie with a fantastic cast, I enjoyed it", "positive"),
    ("The staff were friendly and the rooms were excellent", "positive"),
    ("This game is fun and the graphics are beautiful", "positive"),
    ("I am happy with this purchase, it works great", "positive"),
    ("Wonderful service and a perfect location, highly recommended", "positive"),
    ("I hated this movie, the acting was terrible", "negative"),
    ("What a terrible experience, the staff were rude", "negative"),
    ("The food was cold and the service was terrible", "negative"),
    ("This book is boring, I hated every chapter", "negative"),
    ("The hotel room was dirty and the staff were awful", "negative"),
    ("The course was confusing and the teacher was terrible", "negative"),
    ("Very disappointing show, I was very unhappy", "negative"),
    ("The app crashes constantly and is terrible to use", "negative"),
    ("I had a miserable day stuck in terrible traffic", "negative"),
    ("The new phone is slow, fragile, and overpriced", "negative"),
    ("Poor quality product, I hated it and want a refund", "negative"),
    ("The concert was loud and the sound was terrible", "negative"),
    ("The room was dirty and the breakfast was awful", "negative"),
    ("I hated the trip, the guide was rude and terrible", "negative"),
    ("The dinner was bland and the waiter was awful", "negative"),
    ("Boring movie with a weak cast, I hated it", "negative"),
    ("The staff were rude and the rooms were dirty", "negative"),
    ("This game is boring and the graphics are awful", "negative"),
    ("I am unhappy with this purchase, it works terribly", "negative"),
    ("Terrible service and a poor location, avoid this place", "negative"),
]

NEW_TEXTS = [
    "The service was great and the room was lovely",
    "This was a terrible meal and the staff were awful",
    "The movie was okay but a bit slow",
]


def run_text_classification():
    texts = [text for text, _ in DATASET]
    labels = [label for _, label in DATASET]

    # Split first, so the vectoriser only ever sees training data.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Training samples: {len(X_train_text)}, test samples: {len(X_test_text)}")

    # Feature extraction: text -> TF-IDF numerical vectors (CountVectorizer also works).
    # stop_words removes very common words ('the', 'was') that carry no sentiment.
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # TRAINING: the model learns weights from labelled examples.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # INFERENCE: the frozen model maps unseen inputs to predictions.
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {accuracy:.2f} (single 70/30 split, {len(y_test)} samples)")
    # Cross-validation gives a steadier estimate than one small split.
    cv_scores = cross_val_score(
        make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression(max_iter=1000)),
        texts, labels, cv=5)
    print(f"5-fold cross-validation accuracy: {cv_scores.mean():.2f} (folds: {', '.join(f'{s:.2f}' for s in cv_scores)})")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Predictions for new text (inference only):")
    new_features = vectorizer.transform(NEW_TEXTS)
    new_preds = model.predict(new_features)
    probabilities = model.predict_proba(new_features)
    for text, pred, probs in zip(NEW_TEXTS, new_preds, probabilities):
        confidence = probs.max()
        print(f"  '{text}'\n    -> {pred} (confidence {confidence:.2f})")

    print("\nTraining vs inference:")
    print("- Training fits the vectoriser and model parameters on labelled data (fit).")
    print("- Inference applies the already-trained objects to new data (transform/predict).")
    print("- Inference never updates the model; the same input always gives the same output.")
    return model, vectorizer
