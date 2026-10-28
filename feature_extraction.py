"""
feature_extraction.py
---------------------
Step 4: TF-IDF Feature Extraction for Medical Text Classification.

Demonstrates converting medical text into numerical feature vectors:
1. Loads training and test datasets.
2. Preprocesses abstracts using clean_medical_text.
3. Initializes TfidfVectorizer with unigram + bigram support and English stop words.
4. Fits ONLY on training text and transforms both train and test sets.
5. Displays resulting matrix shapes and examines top TF-IDF features for sample abstracts.
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import clean_medical_text

# Paths
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "medical_tc_train.csv"
TEST_PATH = DATA_DIR / "medical_tc_test.csv"
LABELS_PATH = DATA_DIR / "medical_tc_labels.csv"


def main() -> None:
    print("=" * 75)
    print(" STEP 4: TF-IDF FEATURE EXTRACTION")
    print("=" * 75)

    # 1. Load data
    print("\n[1] Loading dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["condition_label"], labels_df["condition_name"]))

    print(f"Loaded {len(train_df)} training rows and {len(test_df)} test rows.")

    # 2. Preprocess text
    print("\n[2] Preprocessing abstracts...")
    train_cleaned = train_df["medical_abstract"].apply(clean_medical_text)
    test_cleaned = test_df["medical_abstract"].apply(clean_medical_text)
    print("Preprocessed all abstracts successfully.")

    # 3. Configure TfidfVectorizer
    # We use:
    # - max_features=10000: limits vocabulary to top 10,000 most informative terms
    # - ngram_range=(1, 2): captures single words ("cancer") and phrases ("heart failure", "blood pressure")
    # - stop_words='english': filters out general English filler words ("the", "is", "at")
    # - sublinear_tf=True: applies logarithmic scaling 1 + log(tf) to dampen repetitive words
    print("\n[3] Initializing TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    # 4. Fit ONLY on training data, then transform training text
    print("\n[4] Fitting TfidfVectorizer ONLY on training data...")
    X_train_tfidf = vectorizer.fit_transform(train_cleaned)
    print("Fit and transform on training data complete!")

    # 5. Transform test data using the fitted vectorizer (DO NOT FIT on test)
    print("\n[5] Transforming test data (using the already-fitted vectorizer)...")
    X_test_tfidf = vectorizer.transform(test_cleaned)
    print("Transform on test data complete!")

    # 6. Print dimensions and matrix properties
    print("\n" + "=" * 75)
    print(" [6] MATRIX DIMENSIONS & VOCABULARY OVERVIEW")
    print("=" * 75)
    print(f"Total vocabulary features learned from training data : {len(vectorizer.vocabulary_):,}")
    print(f"Training TF-IDF matrix shape (Documents, Features)    : {X_train_tfidf.shape}")
    print(f"Test TF-IDF matrix shape (Documents, Features)        : {X_test_tfidf.shape}")
    print(f"Matrix type                                           : {type(X_train_tfidf).__name__} (sparse matrix)")
    print(f"Training non-zero entries (stored values)             : {X_train_tfidf.nnz:,}")
    sparsity = 100.0 * (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))
    print(f"Matrix sparsity                                       : {sparsity:.2f}% (only non-zero values stored)")

    # 7. Inspect top TF-IDF words for a specific sample abstract
    print("\n" + "=" * 75)
    print(" [7] SAMPLE DEMONSTRATION: TOP TF-IDF FEATURES FOR AN ABSTRACT")
    print("=" * 75)

    sample_idx = 1  # Abstract about pheochromocytoma (Category 1: Neoplasms)
    sample_text = train_cleaned.iloc[sample_idx]
    sample_cat_id = train_df.loc[sample_idx, "condition_label"]
    sample_cat_name = label_map.get(sample_cat_id, "Unknown")

    print(f"Document Index : {sample_idx}")
    print(f"Actual Category: {sample_cat_id} ({sample_cat_name})")
    print(f"Abstract Snippet: \"{sample_text[:140]}...\"\n")

    # Extract words and corresponding TF-IDF scores for this document
    feature_names = vectorizer.get_feature_names_out()
    sample_vector = X_train_tfidf[sample_idx].toarray().flatten()
    top_feature_indices = sample_vector.argsort()[::-1][:10]

    print(f"{'Rank':<6} {'Feature / Term':<25} {'TF-IDF Weight':<15}")
    print("-" * 50)
    for rank, feat_idx in enumerate(top_feature_indices, 1):
        if sample_vector[feat_idx] > 0:
            print(f"{rank:<6} {feature_names[feat_idx]:<25} {sample_vector[feat_idx]:.4f}")

    print("-" * 50)
    print("\nNotice how distinctive clinical terms ('nse', 'pheochromocytoma', 'plasma npy', 'malignant')")
    print("receive the highest mathematical weights, perfectly representing Neoplasms (tumors)!")


if __name__ == "__main__":
    main()
