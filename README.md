# SentimentAI · Restaurant Review Analyzer

A full-stack machine learning application that predicts the sentiment (Positive or Negative) of restaurant reviews. Built with a **FastAPI** backend, an **Ensemble classifier** (Logistic Regression + SVM + Naive Bayes), and a stunning, responsive dark-themed frontend.

This project was initialized using **`uv`** for fast and reliable dependency management.

---

## ✨ Features

- **Real-time Sentiment Prediction:** Enter any restaurant review and instantly get a Positive/Negative prediction with confidence scores.
- **Probability Visualizations:** Animated progress bars display positive vs. negative probability percentages.
- **Model Performance Dashboard:** Live, dynamically loaded metrics including:
  - Test Accuracy (animated donut chart)
  - Precision, Recall, and F1 Score
  - 5-Fold Cross-Validation Accuracy (± standard deviation)
  - Interactive Confusion Matrix (TP / FP / FN / TN)
- **Text Preprocessing Preview:** See exactly how the input text was cleaned, lemmatized, and filtered before being fed to the model.
- **Sample Data Integration:** Click on randomly sampled reviews from the training dataset to quickly test the model.
- **One-Click Retrain:** A dedicated button to clear the cached model and retrain the entire pipeline from scratch.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend / API** | [FastAPI](https://fastapi.tiangolo.com/), Uvicorn |
| **Machine Learning** | scikit-learn (`LogisticRegression`, `LinearSVC`, `MultinomialNB`, `VotingClassifier`, `TfidfVectorizer`) |
| **NLP** | NLTK (`WordNetLemmatizer`, Stopwords), TextBlob |
| **Frontend** | Vanilla HTML5, CSS3 (dark theme, CSS Custom Properties, animations), JavaScript |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) |

---

## 📊 Model Performance (v2 — Improved)

The improved pipeline achieved a significant accuracy boost over the baseline:

| Metric | v1 (Baseline) | v2 (Improved) | Change |
|---|---|---|---|
| **Accuracy** | 77.5% | **84.0%** | +6.5 pts |
| **Precision** | 78.4% | **86.6%** | +8.2 pts |
| **Recall** | 77.7% | **81.6%** | +3.9 pts |
| **F1 Score** | — | **84.0%** | New |
| **CV Accuracy (5-fold)** | — | **82.5% ±2.6%** | New |

### Key improvements made:
1. **TF-IDF with bigrams** instead of raw CountVectorizer — captures phrases like "not good" and "very tasty"
2. **WordNet Lemmatizer** instead of Porter Stemmer — preserves word meaning better (e.g., "better" → "good" via adjective lemmatization)
3. **Sentiment-aware stopword filtering** — keeps negation words ("not", "no", "never", "hardly") and intensifiers ("very", "too") that carry sentiment signal
4. **Contraction expansion** — "isn't" → "is not", "don't" → "do not" before tokenization
5. **Ensemble model (VotingClassifier)** — soft-voting combination of Logistic Regression, LinearSVC, and MultinomialNB with optimized weights (2:2:1)
6. **5-fold Cross-Validation** — a more robust and reliable accuracy estimate

---

## 📁 Project Structure

```text
sentiment_ui/
├── main.py              # FastAPI server — API routes and web UI serving
├── model.py             # ML pipeline — preprocessing, training, prediction
├── pyproject.toml       # uv project configuration and dependencies
├── templates/
│   └── index.html       # The frontend UI (single-page app)
├── static/              # Static assets directory
├── model_cache_v2.pkl   # Auto-generated model cache (created on first run)
└── README.md            # This file
```

> **Note:** The model expects the training data file `Restaurant_Reviews.tsv` to be located in the parent directory (`../Restaurant_Reviews.tsv`).

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Installation

Navigate to the project directory:

```bash
cd /mnt/work/Ai_projects/NLP/semantic_analysis/sentiment_ui
```

`uv` automatically resolves and installs dependencies from `pyproject.toml` when you run the app. To install manually:

```bash
uv sync
```

### 3. Running the Application (Backend & Frontend)

Because this is a full-stack configuration, the FastAPI backend also serves the frontend UI. You only need to run a single server command.

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> On first startup, the model will automatically train and cache to `model_cache_v2.pkl`. Subsequent starts load the cache instantly.

### 4. Open the App (Frontend)

Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser to access the frontend UI.

---

## 🧠 How the ML Pipeline Works

```
Raw Review Text
      │
      ▼
┌─────────────────────────┐
│   1. Preprocessing      │
│   • Expand contractions │
│   • Remove non-alpha    │
│   • Lowercase           │
│   • Remove stopwords    │
│     (keep negations)    │
│   • Lemmatize (adj/verb)│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. TF-IDF Vectorizer   │
│   • Unigrams + Bigrams  │
│   • max 3000 features   │
│   • Sublinear TF        │
│   • min_df=2, max_df=95%│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Ensemble Classifier │
│   • Logistic Regression │
│   • LinearSVC (calib.)  │
│   • MultinomialNB       │
│   • Soft voting (2:2:1) │
└────────────┬────────────┘
             │
             ▼
   Positive / Negative
   + confidence score
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the frontend UI |
| `POST` | `/api/predict` | Predict sentiment for a review (`{"text": "..."}`) |
| `GET` | `/api/metrics` | Get model performance metrics |
| `GET` | `/api/samples` | Get sample reviews from the dataset |
| `POST` | `/api/retrain` | Retrain the model from scratch |
