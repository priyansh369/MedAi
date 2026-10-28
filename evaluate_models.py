"""
evaluate_models.py
------------------
Step 6: Model Evaluation for Medical Text Classification.

Evaluates both trained classifiers on the held-out test dataset (2,888 abstracts):
1. Loads saved models (Naive Bayes & Logistic Regression) and TF-IDF vectorizer.
2. Evaluates predictions against ground truth labels.
3. Computes:
   - Overall Accuracy
   - Macro and Weighted Precision, Recall, and F1-Score
   - Per-class Classification Reports
   - Confusion Matrices with row/column labels
4. Displays a side-by-side comparison table.
5. Exports reports and metrics to CSV and text files for easy use in college reports and slides.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocess import clean_medical_text

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEST_PATH = DATA_DIR / "medical_tc_test.csv"
LABELS_PATH = DATA_DIR / "medical_tc_labels.csv"


def format_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    """Formats confusion matrix with headers and readable alignment."""
    col_header = "True \\ Pred"
    header = f"{col_header:<34}" + "".join([f"[{i+1:>2}] " for i in range(len(class_names))])
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        row_str = f"[{i+1}] {class_names[i]:<30}" + "".join([f"{val:>5} " for val in row])
        lines.append(row_str)
    return "\n".join(lines)


def main() -> None:
    print("=" * 78)
    print(" STEP 6: COMPREHENSIVE MODEL EVALUATION ON HELD-OUT TEST DATA")
    print("=" * 78)

    # 1. Load labels and test data
    print("\n[1] Loading test data and class labels...")
    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["condition_label"], labels_df["condition_name"]))
    class_ids = sorted(label_map.keys())
    class_names = [label_map[i] for i in class_ids]

    test_df = pd.read_csv(TEST_PATH)
    y_test = test_df["condition_label"]
    print(f"Loaded {len(test_df)} test abstracts across {len(class_names)} categories.")

    # 2. Load trained models and vectorizer
    print("\n[2] Loading trained vectorizer and models from 'models/'...")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    nb_model = joblib.load(MODELS_DIR / "naive_bayes_model.joblib")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_model.joblib")
    print("Models and vectorizer loaded successfully.")

    # 3. Preprocess and vectorizer transform
    print("\n[3] Preprocessing and transforming test abstracts...")
    test_cleaned = test_df["medical_abstract"].apply(clean_medical_text)
    X_test_tfidf = vectorizer.transform(test_cleaned)
    print(f"Test TF-IDF matrix shape: {X_test_tfidf.shape}")

    # 4. Generate predictions
    print("\n[4] Generating predictions on held-out test data...")
    nb_preds = nb_model.predict(X_test_tfidf)
    lr_preds = lr_model.predict(X_test_tfidf)

    # 5. Compute metrics
    models_dict = {
        "Multinomial Naive Bayes": nb_preds,
        "Logistic Regression": lr_preds,
    }

    summary_rows = []

    for model_name, preds in models_dict.items():
        acc = accuracy_score(y_test, preds)
        p_macro = precision_score(y_test, preds, average="macro")
        r_macro = recall_score(y_test, preds, average="macro")
        f1_macro = f1_score(y_test, preds, average="macro")

        p_weighted = precision_score(y_test, preds, average="weighted")
        r_weighted = recall_score(y_test, preds, average="weighted")
        f1_weighted = f1_score(y_test, preds, average="weighted")

        summary_rows.append({
            "Model Name": model_name,
            "Accuracy": acc,
            "Macro Precision": p_macro,
            "Macro Recall": r_macro,
            "Macro F1": f1_macro,
            "Weighted Precision": p_weighted,
            "Weighted Recall": r_weighted,
            "Weighted F1": f1_weighted,
        })

    summary_df = pd.DataFrame(summary_rows)

    # 6. Display Comparison Table
    print("\n" + "=" * 78)
    print(" [5] OVERALL PERFORMANCE COMPARISON TABLE")
    print("=" * 78)
    table_view = summary_df.copy()
    for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted Precision", "Weighted Recall", "Weighted F1"]:
        table_view[col] = table_view[col].apply(lambda x: f"{x * 100:.2f}%")
    print(table_view.to_string(index=False))

    # 7. Per-Class Reports & Confusion Matrices
    cm_dict = {}
    for model_name, preds in models_dict.items():
        print("\n" + "=" * 78)
        print(f" DETAILED EVALUATION: {model_name.upper()}")
        print("=" * 78)

        print("\nPer-Class Classification Report:")
        print(classification_report(y_test, preds, target_names=class_names, digits=4))

        cm = confusion_matrix(y_test, preds, labels=class_ids)
        cm_dict[model_name] = cm
        print("Confusion Matrix:")
        print(format_confusion_matrix(cm, class_names))

    # 8. Export to CSV and Text Report for user's PPT and documentation
    summary_csv_path = REPORTS_DIR / "metrics_comparison.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    report_text_path = REPORTS_DIR / "evaluation_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("MEDICAL TEXT CLASSIFICATION USING NLP - MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. OVERALL METRICS COMPARISON:\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        for model_name, preds in models_dict.items():
            f.write(f"\n2. {model_name.upper()} DETAILS:\n")
            f.write(classification_report(y_test, preds, target_names=class_names, digits=4) + "\n")
            f.write("Confusion Matrix:\n")
            f.write(format_confusion_matrix(cm_dict[model_name], class_names) + "\n")

    print("\n" + "=" * 78)
    print(" [6] REPORT EXPORT COMPLETED")
    print("=" * 78)
    print(f">> Saved CSV comparison table : {summary_csv_path}")
    print(f">> Saved full text report     : {report_text_path}")


if __name__ == "__main__":
    main()
