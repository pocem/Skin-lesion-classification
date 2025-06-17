# train_binary_model.py (with SMOTE and Prediction Output)

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import SMOTE from imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("imbalanced-learn library not found. Please install it using: pip install -U imbalanced-learn")
    sys.exit(1)

def train_and_evaluate_binary(binary_dataset_path: str, results_dir: str):
    """
    Loads the binary dataset, applies SMOTE to the training set, performs a 70/15/15 split,
    trains a RandomForestClassifier, evaluates its performance, and saves detailed predictions.
    """
    print("--- Starting Binary Model Training and Evaluation (with SMOTE) ---")

    # --- 1. Load Data ---
    try:
        data_df = pd.read_csv(binary_dataset_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Binary dataset not found at '{binary_dataset_path}'")
        sys.exit(1)

    # --- 2. Data Preparation ---
    print("\nPreparing data for binary modeling...")
    if 'label' not in data_df.columns:
        print("FATAL ERROR: The binary 'label' column was not found in the dataset.")
        sys.exit(1)
        
    potential_non_feature_cols = ['filename', 'label']
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]
    X = data_df[feature_columns]
    y = data_df['label']
    filenames = data_df['filename'] # Keep track of filenames

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # --- 3. Perform 70/15/15 Split (including filenames) ---
    print("\nPerforming 70/15/15 split (Train/Validation/Test)...")
    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y, filenames, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # --- 4. Apply SMOTE to the Training Set ---
    print("\nAddressing class imbalance on the training set with SMOTE...")
    smote = SMOTE(random_state=42)
    print("Original training set distribution:", y_train.value_counts().to_dict())
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled training set distribution:", pd.Series(y_train_resampled).value_counts().to_dict())

    # --- 5. Model Training ---
    print("\nTraining RandomForestClassifier on RESAMPLED data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    print("Training complete.")
    
    # --- 6. Evaluation on Hold-Out Test Set ---
    print("\n--- FINAL EVALUATION ON UNTOUCHED TEST SET (15%) ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    class_names = ['Benign (0)', 'Malignant (1)']
    
    print("\nClassification Report:")
    report_str = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report_str)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=[f"Predicted {c}" for c in class_names])
    print(cm_df)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")

    # --- 7. Create and Save Detailed Prediction DataFrame ---
    print("\nCreating detailed prediction output file...")
    
    predictions_df = pd.DataFrame({
        'filename': f_test.values,
        'true_label': y_test.values,
        'predicted_label': y_pred,
        'proba_benign': y_pred_proba[:, 0],    # Probability of class 0
        'proba_malignant': y_pred_proba[:, 1] # Probability of class 1
    })
    
    # Add a column to easily spot incorrect predictions
    predictions_df['is_correct'] = (predictions_df['true_label'] == predictions_df['predicted_label'])
    
    # --- 8. Save All Results ---
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the detailed predictions
    predictions_path = os.path.join(results_dir, "binary_test_set_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Detailed predictions saved to: {predictions_path}")
    
    # Save the text report
    report_path = os.path.join(results_dir, "binary_classification_report_with_smote.txt")
    with open(report_path, "w") as f:
        f.write(f"--- BINARY CLASSIFICATION RESULTS (with SMOTE) ---\n\n")
        f.write(f"Overall Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
    
    print(f"Evaluation summary saved to: {report_path}")


if __name__ == "__main__":
    # Input for this script
    BINARY_DATASET_PATH = r'C:\Users\misog\portfolio\Machine learning skin lesion project\Skin-lesion-classification\result\final_binary_dataset.csv'
    
    # Where to save the output reports and predictions
    RESULTS_DIRECTORY = './binary_model_results'
    
    # Run the Training and Evaluation
    train_and_evaluate_binary(BINARY_DATASET_PATH, RESULTS_DIRECTORY)