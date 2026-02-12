import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_SETS = ["test", "private_test"]


def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error for regression.
    
    Parameters:
    -----------
    predictions : pd.DataFrame or pd.Series
        Predicted values (can be DataFrame with one column or Series)
    targets : pd.DataFrame or pd.Series
        True target values (can be DataFrame with one column or Series)
    
    Returns:
    --------
    float : Mean Absolute Error
    """
    # Convert to numpy arrays, handling both DataFrame and Series
    if isinstance(predictions, pd.DataFrame):
        pred_values = predictions.iloc[:, 0].values if predictions.shape[1] > 0 else predictions.values.flatten()
    else:
        pred_values = predictions.values
    
    if isinstance(targets, pd.DataFrame):
        target_values = targets.iloc[:, 0].values if targets.shape[1] > 0 else targets.values.flatten()
    else:
        target_values = targets.values
    
    # Handle NaN values - fill with a large error value or remove
    # For MAE, we'll compute on non-NaN pairs only
    mask = ~(np.isnan(pred_values) | np.isnan(target_values))
    if mask.sum() == 0:
        return float('inf')  # All predictions or targets are NaN
    
    pred_clean = pred_values[mask]
    target_clean = target_values[mask]
    
    # Compute MAE: mean(|predictions - targets|)
    mae = np.mean(np.abs(pred_clean - target_clean))
    return float(mae)


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    
    # Verify prediction directory exists
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")
    
    # List files in prediction directory for debugging
    print(f"Prediction directory: {prediction_dir}")
    print(f"Files in prediction directory: {list(prediction_dir.glob('*'))}")
    
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')

        prediction_file = prediction_dir / f'{eval_set}_predictions.csv'
        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {prediction_file}\n"
                f"Available files: {list(prediction_dir.glob('*'))}"
            )
        
        predictions = pd.read_csv(prediction_file)
        print(f"Loaded {len(predictions)} predictions from {prediction_file}")
        
        target_file = reference_dir / eval_set / f'{eval_set}_labels.csv'
        if not target_file.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
        
        targets = pd.read_csv(target_file)
        print(f"Loaded {len(targets)} targets from {target_file}")

        scores[eval_set] = float(compute_mae(predictions, targets))

    # Add train and test times in the score
    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
