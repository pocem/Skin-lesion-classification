import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
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
    Loads binary data, applies SMOTE, finds the best model parameters using GridSearchCV,
    evaluates the best model, reports feature importance, and saves detailed predictions.
    """
    print("--- Starting Advanced Model Training and Tuning Pipeline ---")

    # --- 1. Load Data ---
    try:
        data_df = pd.read_csv(binary_dataset_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Binary dataset not found at '{binary_dataset_path}'")
        sys.exit(1)

    # --- 2. Data Preparation ---
    print("\nPreparing data for binary modeling...")
    potential_non_feature_cols = ['filename', 'label']
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]
    X = data_df[feature_columns]
    y = data_df['label']
    filenames = data_df['filename']

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # --- 3. Perform 70/15/15 Split ---
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
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled training set distribution:", pd.Series(y_train_resampled).value_counts().to_dict())

    # --- 5. Hyperparameter Tuning with GridSearchCV ---
    print("\nPerforming Hyperparameter Tuning to find the best model...")

    # Define a smaller, faster grid for initial tuning. Can be expanded later.
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    
    # We use the resampled training data for the final fit and tuning.
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=3,       # 3-fold cross-validation on the training data
                               n_jobs=-1,  # Use all available CPU cores
                               verbose=2,  # Show progress
                               scoring='f1_weighted') # Optimize for a balanced F1 score

    # Fit the grid search to find the best parameters
    grid_search.fit(X_train_resampled, y_train_resampled)

    print("\nBest parameters found by GridSearchCV:")
    print(grid_search.best_params_)
    
    # The best model is automatically refit on the entire training data
    best_model = grid_search.best_estimator_

    # --- 6. Evaluation on Hold-Out Test Set using the BEST model ---
    print("\n--- FINAL EVALUATION ON UNTOUCHED TEST SET (15%) ---")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    class_names = ['Benign (0)', 'Malignant (1)']
    
    print("\nClassification Report (from best model):")
    report_str = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report_str)
    
    print("\nConfusion Matrix (from best model):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=[f"Predicted {c}" for c in class_names])
    print(cm_df)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    
    # --- 7. Feature Importance Analysis ---
    print("\n--- FEATURE IMPORTANCE ANALYSIS ---")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(feature_importance_df.head(15))

    # --- 8. Save All Results ---
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'filename': f_test.values, 'true_label': y_test.values, 'predicted_label': y_pred,
        'proba_benign': y_pred_proba[:, 0], 'proba_malignant': y_pred_proba[:, 1]
    })
    predictions_df['is_correct'] = (predictions_df['true_label'] == predictions_df['predicted_label'])
    predictions_path = os.path.join(results_dir, "binary_test_set_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nDetailed predictions saved to: {predictions_path}")
    
    # Save the text report
    report_path = os.path.join(results_dir, "binary_classification_report_tuned.txt")
    with open(report_path, "w") as f:
        f.write(f"--- BINARY CLASSIFICATION RESULTS (Tuned with GridSearchCV + SMOTE) ---\n\n")
        f.write(f"Best Hyperparameters Found:\n{grid_search.best_params_}\n\n")
        f.write(f"Overall Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nFeature Importances:\n")
        f.write(feature_importance_df.to_string())
    
    print(f"Full evaluation report saved to: {report_path}")

if __name__ == "__main__":
    # Input the dataset that was built for supervised machine learning in this script
    DATASET_PATH = r''
    
    # Where to save the output reports and predictions
    RESULTS_DIRECTORY = r''
    
    train_and_evaluate_binary(DATASET_PATH, RESULTS_DIRECTORY)