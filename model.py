"""
model.py — Sentiment Analysis ML Pipeline
Handles text preprocessing, model training, caching, and prediction.
"""

import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ── NLTK data ─────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR.parent / "Restaurant_Reviews.tsv"
MODEL_CACHE = BASE_DIR / "model_cache_v2.pkl"

# ── Text preprocessing ───────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

# Keep sentiment-critical stopwords
STOP_WORDS = set(stopwords.words("english"))
KEEP_WORDS = {
    "not", "no", "nor", "never", "neither", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "don", "dont", "doesn",
    "doesn't", "don't", "isn", "isn't", "wasn", "wasn't", "weren",
    "weren't", "hasn", "hasn't", "haven", "haven't", "hasn", "hadn",
    "hadn't", "couldn", "couldn't", "wouldn", "wouldn't", "shouldn",
    "shouldn't", "won", "won't", "but", "very", "too", "only",
    "against", "above", "below",
}
STOP_WORDS -= KEEP_WORDS


def preprocess(text: str) -> str:
    """Clean, lemmatize and selectively remove stopwords from review text."""
    text = str(text).lower()
    # Expand contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    # Remove non-alphabetical characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # Tokenize
    tokens = text.split()
    # Lemmatize + filter stopwords (keep sentiment words)
    processed = []
    for w in tokens:
        if len(w) <= 2 and w not in KEEP_WORDS:
            continue
        if w in STOP_WORDS:
            continue
        lemma_v = lemmatizer.lemmatize(w, pos="v")
        lemma_n = lemmatizer.lemmatize(w, pos="a")
        lemma = min(lemma_v, lemma_n, key=len)
        processed.append(lemma)
    return " ".join(processed)


def _load_dataframe() -> pd.DataFrame:
    """Load the restaurant reviews dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")
    return pd.read_csv(DATA_FILE, delimiter="\t", quoting=3)


def _build_ensemble():
    """Build the ensemble classifier."""
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=0),
        cv=3,
    )
    nb = MultinomialNB(alpha=0.5)
    return VotingClassifier(
        estimators=[("lr", lr), ("svm", svm), ("nb", nb)],
        voting="soft",
        weights=[2, 2, 1],
    )


def _build_vectorizer():
    """Build the TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def _compute_metrics(y_test, y_pred, labels, X, ensemble) -> dict:
    """Compute all evaluation metrics."""
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(ensemble, X, labels, cv=5, scoring="accuracy")
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred)) * 100, 2),
        "f1": round(float(f1_score(y_test, y_pred)) * 100, 2),
        "cv_accuracy": round(float(cv_scores.mean()) * 100, 2),
        "cv_std": round(float(cv_scores.std()) * 100, 2),
        "total_samples": int(len(labels)),
        "train_samples": int(len(y_test) * 4),  # 80/20 split
        "test_samples": int(len(y_test)),
        "positive_reviews": int((labels == 1).sum()),
        "negative_reviews": int((labels == 0).sum()),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
    }


def _get_sample_reviews(df: pd.DataFrame, n: int = 8, seed: int = 42) -> list[dict]:
    """Get random sample reviews for the UI."""
    sample = df.sample(n, random_state=seed)
    return [
        {"text": row["Review"], "label": int(row["Liked"])}
        for _, row in sample.iterrows()
    ]


# ── Public API ────────────────────────────────────────────────────────────────

class SentimentModel:
    """Encapsulates the trained model, vectorizer, metrics, and sample data."""

    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.metrics: dict | None = None
        self.sample_reviews: list[dict] = []
        self.trained: bool = False

    def train(self) -> dict:
        """Train the sentiment classifier from scratch."""
        df = _load_dataframe()
        corpus = [preprocess(r) for r in df["Review"]]
        labels = df["Liked"].values

        self.vectorizer = _build_vectorizer()
        X = self.vectorizer.fit_transform(corpus)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.20, random_state=0
        )

        self.model = _build_ensemble()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.metrics = _compute_metrics(y_test, y_pred, labels, X, self.model)
        self.sample_reviews = _get_sample_reviews(df)
        self.trained = True

        # Cache to disk
        with open(MODEL_CACHE, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "model": self.model,
                "metrics": self.metrics,
            }, f)

        return self.metrics

    def load_or_train(self) -> dict:
        """Load cached model or train fresh."""
        if MODEL_CACHE.exists():
            try:
                with open(MODEL_CACHE, "rb") as f:
                    cached = pickle.load(f)
                self.vectorizer = cached["vectorizer"]
                self.model = cached["model"]
                self.metrics = cached["metrics"]
                self.trained = True
                # Load samples fresh
                if DATA_FILE.exists():
                    df = _load_dataframe()
                    self.sample_reviews = _get_sample_reviews(df)
                return self.metrics
            except Exception:
                pass  # retrain if cache is bad
        return self.train()

    def retrain(self) -> dict:
        """Clear cache and retrain."""
        if MODEL_CACHE.exists():
            MODEL_CACHE.unlink()
        return self.train()

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single review."""
        if not self.trained:
            raise RuntimeError("Model is not trained yet.")

        processed = preprocess(text)
        if not processed.strip():
            raise ValueError("Review text is empty after cleaning.")

        vec = self.vectorizer.transform([processed])
        proba = self.model.predict_proba(vec)[0]
        pred = self.model.predict(vec)[0]

        neg_prob, pos_prob = float(proba[0]), float(proba[1])
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = pos_prob if pred == 1 else neg_prob

        return {
            "sentiment": sentiment,
            "confidence": round(confidence * 100, 1),
            "positive_prob": round(pos_prob * 100, 1),
            "negative_prob": round(neg_prob * 100, 1),
            "processed_text": processed,
        }
