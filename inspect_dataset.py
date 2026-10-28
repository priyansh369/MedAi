"""
inspect_dataset.py
------------------
Step 2: Dataset Verification and Inspection Script.
Performs data validation on the Medical Abstracts Text Classification Corpus:
1. Verifies file presence and header rows.
2. Identifies label and text columns.
3. Inspects shapes, data types, missing values, and sample records.
4. Validates class distributions against official documentation counts.
"""

from pathlib import Path
import pandas as pd

# Define paths
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "medical_tc_train.csv"
TEST_PATH = DATA_DIR / "medical_tc_test.csv"
LABELS_PATH = DATA_DIR / "medical_tc_labels.csv"

# Expected counts from official dataset documentation (README.md)
EXPECTED_COUNTS = {
    1: {"name": "Neoplasms", "train": 2530, "test": 633},
    2: {"name": "Digestive system diseases", "train": 1195, "test": 299},
    3: {"name": "Nervous system diseases", "train": 1540, "test": 385},
    4: {"name": "Cardiovascular diseases", "train": 2441, "test": 610},
    5: {"name": "General pathological conditions", "train": 3844, "test": 961},
}


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_file_headers() -> None:
    print_section("1. FILE PRESENCE & HEADER CHECK")
    for name, path in [("Training file", TRAIN_PATH), ("Test file", TEST_PATH), ("Labels file", LABELS_PATH)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        print(f"[{name}] {path.name}")
        print(f"  First line (header row) : {first_line}")
        print(f"  First data sample       : {second_line[:80]}...\n")


def inspect_dataframe(df: pd.DataFrame, split_name: str, label_map: dict) -> None:
    print_section(f"2. {split_name.upper()} DATASET OVERVIEW")
    print(f"Shape (Rows, Columns): {df.shape}")
    print(f"Column Names: {list(df.columns)}")

    print("\nMissing Values per Column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        print(f"  - {col}: {count} missing")

    print("\nFirst 5 Rows:")
    for idx, row in df.head(5).iterrows():
        label_id = row["condition_label"]
        label_name = label_map.get(label_id, "Unknown")
        abstract_snippet = str(row["medical_abstract"])[:110].strip().replace("\n", " ")
        print(f"  [{idx + 1}] Label {label_id} ({label_name}):")
        print(f"      \"{abstract_snippet}...\"")


def verify_category_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, label_map: dict) -> None:
    print_section("3. CATEGORY DISTRIBUTION & VERIFICATION AGAINST DOCUMENTATION")
    train_counts = train_df["condition_label"].value_counts().to_dict()
    test_counts = test_df["condition_label"].value_counts().to_dict()

    print(f"{'ID':<4} {'Condition Name':<34} {'Train':<10} {'Exp.':<8} {'Test':<10} {'Exp.':<8} {'Status'}")
    print("-" * 85)

    all_matched = True
    for label_id in sorted(EXPECTED_COUNTS.keys()):
        expected = EXPECTED_COUNTS[label_id]
        name = label_map.get(label_id, expected["name"])
        actual_train = train_counts.get(label_id, 0)
        actual_test = test_counts.get(label_id, 0)

        match = (actual_train == expected["train"]) and (actual_test == expected["test"])
        if not match:
            all_matched = False

        status = "MATCH [OK]" if match else "MISMATCH [!]"
        print(f"{label_id:<4} {name:<34} {actual_train:<10} {expected['train']:<8} {actual_test:<10} {expected['test']:<8} {status}")

    print("-" * 85)
    total_train = len(train_df)
    total_test = len(test_df)
    print(f"{'Total':<39} {total_train:<10} {11550:<8} {total_test:<10} {2888:<8}")

    if all_matched:
        print("\nSUCCESS: All 5 categories perfectly match the official dataset counts!")
    else:
        print("\nWARNING: Some counts did not match the documentation.")


def main() -> None:
    # 1. Check file headers
    check_file_headers()

    # 2. Load labels mapping
    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["condition_label"], labels_df["condition_name"]))

    print_section("IDENTIFIED COLUMNS")
    print(f"- Label Column : 'condition_label' (Numeric ID: 1 through 5)")
    print(f"- Text Column  : 'medical_abstract' (Full research paper text)")
    print("\nClass Mapping:")
    for cid, cname in sorted(label_map.items()):
        print(f"  {cid} -> {cname}")

    # 3. Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 4. Inspect train and test sets
    inspect_dataframe(train_df, "Training Set", label_map)
    inspect_dataframe(test_df, "Test Set", label_map)

    # 5. Verify category distribution
    verify_category_distribution(train_df, test_df, label_map)


if __name__ == "__main__":
    main()
