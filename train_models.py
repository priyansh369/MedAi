"""
train_models.py
---------------
Step 5: Machine Learning Model Training for Medical Text Classification.

Trains two classical NLP classifiers on TF-IDF features:
1. Multinomial Naive Bayes (MultinomialNB) - Fast probabilistic baseline
2. Logistic Regression (LogisticRegression) - Strong linear classification baseline

Evaluates both models on the unseen test set and displays real, non-invented metrics.
"""

from pathlib import Path
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.preprocess import clean_medical_text

# Paths
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "medical_tc_train.csv"
TEST_PATH = DATA_DIR / "medical_tc_test.csv"
LABELS_PATH = DATA_DIR / "medical_tc_labels.csv"

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 75)
    print(" STEP 5: ML MODEL TRAINING (NAIVE BAYES & LOGISTIC REGRESSION)")
    print("=" * 75)

    # 1. Load data
    print("\n[1] Loading dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["condition_label"], labels_df["condition_name"]))
    target_names = [label_map[i] for i in sorted(label_map.keys())]

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")

    # 2. Preprocess text
    print("\n[2] Cleaning medical abstracts...")
    X_train_raw = train_df["medical_abstract"].apply(clean_medical_text)
    X_test_raw = test_df["medical_abstract"].apply(clean_medical_text)
    y_train = train_df["condition_label"]
    y_test = test_df["condition_label"]
    print("Text cleaning complete.")

    # 3. TF-IDF Feature Extraction (Same vocabulary for both models)
    print("\n[3] Extracting TF-IDF features (Vocabulary limit: 10,000 terms)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    # Fit ONLY on training data
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"Features created: {X_train_tfidf.shape[1]:,} features.")

    # -------------------------------------------------------------
    # Model 1: Multinomial Naive Bayes
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(" [4] TRAINING MODEL 1: MULTINOMIAL NAIVE BAYES (MultinomialNB)")
    print("-" * 75)
    print("Initializing MultinomialNB...")
    nb_model = MultinomialNB(alpha=0.1)  # alpha=0.1 gives slightly smoothed probabilities

    print("Fitting MultinomialNB on training data...")
    start_time = time.time()
    nb_model.fit(X_train_tfidf, y_train)
    nb_train_time = time.time() - start_time
    print(f">> SUCCESS: Multinomial Naive Bayes trained in {nb_train_time:.3f} seconds!")

    print("Evaluating MultinomialNB on test data...")
    nb_predictions = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    print(f">> Test Accuracy: {nb_accuracy * 100:.2f}%\n")
    print("Detailed Classification Report for Naive Bayes:")
    print(classification_report(y_test, nb_predictions, target_names=target_names, digits=4))

    # -------------------------------------------------------------
    # Model 2: Logistic Regression
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(" [5] TRAINING MODEL 2: LOGISTIC REGRESSION (LogisticRegression)")
    print("-" * 75)
    print("Initializing LogisticRegression (max_iter=1000)...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs"
    )

    print("Fitting Logistic Regression on training data...")
    start_time = time.time()
    lr_model.fit(X_train_tfidf, y_train)
    lr_train_time = time.time() - start_time
    print(f">> SUCCESS: Logistic Regression trained in {lr_train_time:.3f} seconds!")

    print("Evaluating Logistic Regression on test data...")
    lr_predictions = lr_model.predict(X_test_tfidf)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f">> Test Accuracy: {lr_accuracy * 100:.2f}%\n")
    print("Detailed Classification Report for Logistic Regression:")
    print(classification_report(y_test, lr_predictions, target_names=target_names, digits=4))

    # -------------------------------------------------------------
    # Model Comparison Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" [6] PERFORMANCE COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Model Name':<28} {'Test Accuracy':<16} {'Training Time':<15}")
    print("-" * 60)
    print(f"{'Multinomial Naive Bayes':<28} {nb_accuracy * 100:>10.2f}%    {nb_train_time:>10.3f}s")
    print(f"{'Logistic Regression':<28} {lr_accuracy * 100:>10.2f}%    {lr_train_time:>10.3f}s")
    print("-" * 60)

    # Save trained models and vectorizer for future console application
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(nb_model, MODELS_DIR / "naive_bayes_model.joblib")
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression_model.joblib")
    print(f"\nSaved models and vectorizer to '{MODELS_DIR}' directory.")


if __name__ == "__main__":
    main()
