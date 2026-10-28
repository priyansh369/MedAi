"""
preprocess_text.py
------------------
Step 3: Text Preprocessing for Medical Text Classification.

Demonstrates beginner-friendly text cleaning tailored for medical abstracts:
1. Loads verified dataset (train and test).
2. Checks and handles missing/empty text values.
3. Cleans text safely:
   - Lowercases text for uniform token matching.
   - Cleans brackets, quotes, and punctuation while preserving medical terms (e.g., alphanumeric terms, hyphens).
   - Normalizes whitespace.
4. Compares original vs. cleaned text side-by-side with before-and-after metrics.
"""

from pathlib import Path
import re
import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "medical_tc_train.csv"
TEST_PATH = DATA_DIR / "medical_tc_test.csv"
LABELS_PATH = DATA_DIR / "medical_tc_labels.csv"


def clean_medical_text(text: str) -> str:
    """
    Cleans a single medical abstract while preserving medical vocabulary.

    Medical domain considerations:
    - Keep alphanumeric characters (e.g., 'cd4', 'p53', 'stage2', '150mg').
    - Preserve intra-word hyphens (e.g., 'anti-inflammatory', 'interleukin-1').
    - Remove noisy punctuation: quotes, brackets, semicolons, asterisks, URLs.
    - Normalize whitespace.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase text
    cleaned = text.lower()

    # 2. Remove URLs if any exist
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 3. Replace brackets, parentheses, slashes, and quotes with spaces
    # Example: "(NSE)" -> " nse ", "in vivo/in vitro" -> "in vivo in vitro"
    cleaned = re.sub(r'[\(\)\[\]\{\}\"\'\\\/<>,;:!+*?=~_]', " ", cleaned)

    # 4. Remove standalone hyphens or trailing hyphens, but keep words like 'anti-inflammatory'
    cleaned = re.sub(r"(?<!\w)-|-(?!\w)", " ", cleaned)

    # 5. Remove standalone periods (keep decimal numbers like 1.5 if needed, or remove end-of-sentence periods)
    # This removes periods not surrounded by digits (e.g., "word." -> "word", but keeps "1.5")
    cleaned = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", cleaned)

    # 6. Normalize multiple spaces and linebreaks into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def main() -> None:
    print("=" * 75)
    print(" STEP 3: MEDICAL TEXT PREPROCESSING")
    print("=" * 75)

    # 1. Load data
    print("\n[1] Loading dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["condition_label"], labels_df["condition_name"]))

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")

    # 2. Check and handle missing / empty texts
    print("\n[2] Checking for missing, null, or whitespace-only records...")
    train_missing = train_df["medical_abstract"].isna().sum()
    test_missing = test_df["medical_abstract"].isna().sum()

    print(f"  - Training null values: {train_missing}")
    print(f"  - Test null values: {test_missing}")

    # Fill any potential nulls with empty string as a safeguard
    train_df["medical_abstract"] = train_df["medical_abstract"].fillna("")
    test_df["medical_abstract"] = test_df["medical_abstract"].fillna("")

    # Check for empty or whitespace-only strings
    empty_train = (train_df["medical_abstract"].str.strip() == "").sum()
    empty_test = (test_df["medical_abstract"].str.strip() == "").sum()
    print(f"  - Empty/blank training abstracts: {empty_train}")
    print(f"  - Empty/blank test abstracts: {empty_test}")

    # 3. Apply cleaning function
    print("\n[3] Applying clean_medical_text() to abstracts...")
    train_df["cleaned_abstract"] = train_df["medical_abstract"].apply(clean_medical_text)
    test_df["cleaned_abstract"] = test_df["medical_abstract"].apply(clean_medical_text)
    print("  Preprocessing complete!")

    # 4. Display Side-by-Side Comparisons for 3 distinct examples
    print("\n" + "=" * 75)
    print(" [4] SIDE-BY-SIDE EXAMPLES: ORIGINAL VS. CLEANED TEXT")
    print("=" * 75)

    sample_indices = [0, 1, 3]  # Pick representative medical abstracts
    for i, idx in enumerate(sample_indices, 1):
        label_id = train_df.loc[idx, "condition_label"]
        category = label_map.get(label_id, "Unknown")
        original = train_df.loc[idx, "medical_abstract"]
        cleaned = train_df.loc[idx, "cleaned_abstract"]

        print(f"\n--- Example {i} | Category: {label_id} - {category} ---")
        print("\n[ORIGINAL (First 250 characters)]:")
        print(f"\"{original[:250]}...\"")
        print(f"  -> Total characters: {len(original)} | Total words: {len(original.split())}")

        print("\n[CLEANED (First 250 characters)]:")
        print(f"\"{cleaned[:250]}...\"")
        print(f"  -> Total characters: {len(cleaned)} | Total words: {len(cleaned.split())}")

    # 5. Summary statistics
    print("\n" + "=" * 75)
    print(" [5] SUMMARY METRICS BEFORE & AFTER CLEANING")
    print("=" * 75)
    avg_orig_words = train_df["medical_abstract"].apply(lambda t: len(t.split())).mean()
    avg_clean_words = train_df["cleaned_abstract"].apply(lambda t: len(t.split())).mean()
    print(f"  Average word count per abstract before cleaning : {avg_orig_words:.2f} words")
    print(f"  Average word count per abstract after cleaning  : {avg_clean_words:.2f} words")
    print(f"  Difference                                      : {avg_orig_words - avg_clean_words:.2f} words (punctuation & noise removed)")


if __name__ == "__main__":
    main()
