# Script to load the TiVA dataset and create the splits for Codabench
from pathlib import Path

import pandas as pd

PHASE = 'dev_phase'

# Paths to the CSV files (relative to the root directory)
ROOT_DIR = Path(__file__).parent.parent.parent
# Data directories should be relative to the datachallenge_codabench directory
BUNDLE_DIR = Path(__file__).parent.parent
DATA_DIR = BUNDLE_DIR / PHASE / 'input_data'
REF_DIR = BUNDLE_DIR / PHASE / 'reference_data'
TRAIN_CSV = ROOT_DIR / 'train.csv'
TEST_CSV = ROOT_DIR / 'test.csv'
SOLUTION_CSV = ROOT_DIR / 'solution.csv'


def make_csv(data, filepath):
    """Save data to CSV file, creating directories if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(filepath, index=False)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Load TiVA data and create splits for Codabench'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (not used but kept for compatibility)')
    args = parser.parse_args()

    print("Loading data files...")
    
    # Load training data (contains target)
    print(f"Loading {TRAIN_CSV}...")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"Train shape: {df_train.shape}")
    
    # Load test data (without target) and solution (target only)
    print(f"Loading {TEST_CSV}...")
    df_test = pd.read_csv(TEST_CSV)
    print(f"Test shape: {df_test.shape}")
    
    print(f"Loading {SOLUTION_CSV}...")
    df_solution = pd.read_csv(SOLUTION_CSV)
    print(f"Solution shape: {df_solution.shape}")
    
    # Verify that test and solution have the same number of rows
    assert len(df_test) == len(df_solution), \
        f"Mismatch: test has {len(df_test)} rows but solution has {len(df_solution)} rows"
    
    # Separate features and target for training data
    target_col = 'TiVA_Value_Target'
    assert target_col in df_train.columns, f"Target column '{target_col}' not found in train data"
    
    # Identify feature columns (exclude target and some metadata columns)
    # Exclude categorical string columns that need encoding (participants can add encoding in their model)
    # For now, we exclude them to avoid errors with scikit-learn models that expect numeric data
    exclude_cols = [
        target_col, 
        'country_x', 'country_y',  # Redundant country names
        'Source_Country', 'Target_Country',  # ISO codes (categorical strings)
        'Sector_Code', 'Sector_Name',  # Sector codes/names (categorical strings)
    ]
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    # Verify we have numeric features
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")
    
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy()
    
    print(f"Train features: {X_train.shape}, Train target: {y_train.shape}")
    
    # For test data, separate by year: 2018 = test (public), 2019 = private_test
    print("Splitting test data by year...")
    assert 'Year' in df_test.columns, "Year column not found in test data"
    
    df_test_2018 = df_test[df_test['Year'] == 2018].copy()
    df_test_2019 = df_test[df_test['Year'] == 2019].copy()
    
    # Get corresponding solution rows
    # We need to match the indices - solution.csv should be in the same order as test.csv
    test_indices_2018 = df_test[df_test['Year'] == 2018].index
    test_indices_2019 = df_test[df_test['Year'] == 2019].index
    
    y_test_2018 = df_solution.iloc[test_indices_2018][target_col].copy()
    y_test_2019 = df_solution.iloc[test_indices_2019][target_col].copy()
    
    print(f"Test 2018 (public): {len(df_test_2018)} samples")
    print(f"Test 2019 (private): {len(df_test_2019)} samples")
    
    # Extract features for test sets (same columns as train)
    X_test_2018 = df_test_2018[feature_cols].copy()
    X_test_2019 = df_test_2019[feature_cols].copy()
    
    # Store the data in the correct folders:
    # - input_data contains train data (both features and labels) and only
    #   test features so the test labels are kept secret
    # - reference_data contains the test labels for scoring
    
    print("\nSaving data to Codabench structure...")
    
    # Training data
    make_csv(X_train, DATA_DIR / 'train' / 'train_features.csv')
    make_csv(y_train, DATA_DIR / 'train' / 'train_labels.csv')
    print(f"Saved train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Test data (public - 2018)
    make_csv(X_test_2018, DATA_DIR / 'test' / 'test_features.csv')
    make_csv(y_test_2018, REF_DIR / 'test' / 'test_labels.csv')
    print(f"Saved test (2018): {X_test_2018.shape[0]} samples")
    
    # Private test data (2019)
    make_csv(X_test_2019, DATA_DIR / 'private_test' / 'private_test_features.csv')
    make_csv(y_test_2019, REF_DIR / 'private_test' / 'private_test_labels.csv')
    print(f"Saved private_test (2019): {X_test_2019.shape[0]} samples")
    
    print("\nData preparation complete!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Reference directory: {REF_DIR}")
