import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shutil
from tqdm import tqdm

# Import REFINED custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder
    from util.feature_C import extract_feature_C
    from util.blue_veil import extract_blue_veil_features
except ImportError as e:
    print(f"Error: Could not import refined custom feature modules: {e}")
    sys.exit(1)

def normalize_filename(filename: str) -> str:
    """
    A centralized function to guarantee a standard filename format.
    Removes extensions and common suffixes like '_mask'.
    Example: 'PAT_123_mask.png' -> 'PAT_123'
             'PAT_456.jpg'      -> 'PAT_456'
    """
    # Get the filename without the path
    base = os.path.basename(filename)
    # Get the filename without the extension
    name_without_ext = os.path.splitext(base)[0]
    # Remove the '_mask' suffix if it exists
    cleaned_name = name_without_ext.replace('_mask', '')
    return cleaned_name

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None, recreate_features=False):
    print("Starting REFINED feature extraction process...")

    dfs = {}
    # Define which directory each feature extractor should use
    feature_extractors = {
        "A_asymmetry": (extract_asymmetry_features, {'folder_path': mask_img_dir}),
        "B_border": (extract_border_features_from_folder, {'folder_path': original_img_dir}),
        "C_color": (extract_feature_C, {'folder_path': original_img_dir}),
        "BV_blue_veil": (extract_blue_veil_features, {'folder_path': original_img_dir})
    }

    for name, (func, params) in feature_extractors.items():
        print(f"\nExtracting {name} features...")
        try:
            temp_df = func(**params) # Simplified call
            dfs[name] = temp_df
        except Exception as e:
            print(f"Error during {name} extraction: {e}")
            dfs[name] = pd.DataFrame() # Create empty df on error

    # --- Centralized Filename Normalization ---
    print("\n--- Standardizing all filenames centrally ---")
    for name, df in dfs.items():
        if not df.empty and 'filename' in df.columns:
            print(f"Normalizing filenames for {name}...")
            # Apply our single, robust normalization function
            df['filename'] = df['filename'].apply(normalize_filename)
            print(f"  Sample after normalization: {df['filename'].iloc[0]}")
        else:
            print(f"Warning: DataFrame for {name} is empty. Skipping normalization.")

    # --- Metadata Loading and Normalization ---
    metadata_df = None
    if labels_csv and exists(labels_csv):
        print("\nLoading and normalizing metadata...")
        try:
            metadata_df = pd.read_csv(labels_csv)
            # The 'diagnostic' column will become our 'real_label'
            metadata_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
            
            # CRUCIAL: Also normalize the metadata's filename column using the SAME function
            if 'filename' in metadata_df.columns:
                 metadata_df['filename'] = metadata_df['filename'].apply(normalize_filename)
                 print(f"  Metadata 'filename' sample after normalization: {metadata_df['filename'].iloc[0]}")
                 # Keep only the columns we absolutely need
                 metadata_df = metadata_df[['filename', 'real_label']].copy()
            else:
                print("ERROR: 'filename' column not found in metadata CSV!")
                metadata_df = None

        except Exception as e:
            print(f"Error processing metadata: {e}")

    # --- Merging DataFrames ---
    print("\nMerging all feature DataFrames...")
    
    # Start with metadata if it exists
    final_df = metadata_df
    
    # Iteratively merge each feature dataframe
    for name, df in dfs.items():
        if df.empty or 'filename' not in df.columns:
            print(f"Skipping merge for empty/invalid DataFrame: {name}")
            continue
        
        if final_df is None: # If metadata failed, start with the first feature df
            final_df = df
            print(f"Starting merge with {name} features.")
        else:
            print(f"Merging with {name} features...")
            final_df = pd.merge(final_df, df, on='filename', how='inner')
            print(f"  Shape after merge: {final_df.shape}")
            if final_df.empty:
                print(f"CRITICAL: Merge with {name} resulted in an empty DataFrame. Halting.")
                break

    if final_df is None or final_df.empty:
        print("Merge failed or resulted in an empty DataFrame. Please check normalization logs.")
        return pd.DataFrame()

    print(f"\nMerged DataFrame final shape: {final_df.shape}")
    final_df.to_csv(output_csv_path, index=False)
    print(f"Merged feature dataset saved to {output_csv_path}")
    return final_df

# The rest of your main function (data prep, CV loop) is correct and does not need to change.
# I am including it here for a complete, runnable file.
def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- MULTI-CLASS REFINED FEATURE PIPELINE & MODEL EVALUATION ---\n")

    data_df = None
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new refined feature dataset at {output_csv_path}...")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path,
                                         labels_csv=labels_csv_path, recreate_features=recreate_features)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        data_df = pd.read_csv(output_csv_path)

    if data_df is None or data_df.empty:
        print("Failed to create or load the feature dataset. Exiting."); return

    print("\n--- Data Cleaning and Preparation for Multi-Class ---")
    data_df.drop_duplicates(subset=['filename'], keep='first', inplace=True)
    data_df.dropna(subset=['real_label'], inplace=True)

    unique_labels = sorted(data_df['real_label'].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    data_df['label'] = data_df['real_label'].map(label_to_int)
    class_names_for_report = list(label_to_int.keys())
    
    print("Multi-class label mapping created:")
    print(label_to_int)
    print(f"Label distribution:\n{data_df['real_label'].value_counts(normalize=True)}")
    
    potential_non_feature_cols = ['filename', 'real_label', 'label', 'diagnostic']
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]

    if not feature_columns:
        print("CRITICAL ERROR: No feature columns identified."); return
    print(f"\nUsing {len(feature_columns)} refined feature columns for training.")

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy()
    current_filenames = data_df['filename'].copy()
    
    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns)

    N_SPLITS = 5
    min_class_count = data_df['label'].value_counts().min()
    if N_SPLITS > min_class_count:
        print(f"Warning: N_SPLITS ({N_SPLITS}) > smallest class ({min_class_count}). Reducing to {min_class_count}.")
        N_SPLITS = min_class_count
        if N_SPLITS < 2:
             print("Smallest class has < 2 samples. Cannot perform K-Fold CV."); return

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results_list = []
    all_test_predictions_df = pd.DataFrame()

    print(f"\n--- {N_SPLITS}-FOLD MULTI-CLASS CROSS-VALIDATION ---")
    for fold_num, (dev_indices, test_indices) in enumerate(skf.split(x_all, y_all)):
        print(f"\n--- FOLD {fold_num + 1}/{N_SPLITS} ---")
        x_dev_fold, y_dev_fold = x_all.iloc[dev_indices], y_all.iloc[dev_indices]
        x_test_fold, y_test_fold = x_all.iloc[test_indices], y_all.iloc[test_indices]
        filenames_test_fold = current_filenames.iloc[test_indices]

        try:
            x_train_inner, x_val_inner, y_train_inner, y_val_inner = train_test_split(
                x_dev_fold, y_dev_fold, test_size=0.25, random_state=42, stratify=y_dev_fold
            )
        except ValueError as e:
            print(f"Error during inner split for fold {fold_num + 1}: {e}. Skipping."); continue

        try:
            model_fold = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model_fold.fit(x_train_inner, y_train_inner)
            
            y_test_pred_fold = model_fold.predict(x_test_fold)
            y_test_pred_proba_fold = model_fold.predict_proba(x_test_fold)
            
            print(f"Fold {fold_num + 1} - Test Results:")
            print(classification_report(y_test_fold, y_test_pred_fold, target_names=class_names_for_report, zero_division=0))
            
            cls_report_dict = classification_report(y_test_fold, y_test_pred_fold, target_names=class_names_for_report, output_dict=True, zero_division=0)
            fold_summary = {'fold': fold_num + 1, 'overall_accuracy': cls_report_dict['accuracy']}
            for label, metrics in cls_report_dict.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        fold_summary[f'{label.replace(" ", "_")}_{metric_name}'] = value
            fold_results_list.append(fold_summary)
            
            true_labels_text = y_test_fold.map(int_to_label)
            pred_labels_text = pd.Series(y_test_pred_fold).map(int_to_label)
            
            current_fold_predictions = pd.DataFrame({
                'fold': fold_num + 1, 'filename': filenames_test_fold.values,
                'true_label': true_labels_text.values, 'predicted_label': pred_labels_text.values
            })
            for i, class_name in enumerate(class_names_for_report):
                current_fold_predictions[f'proba_{class_name}'] = y_test_pred_proba_fold[:, i]

            all_test_predictions_df = pd.concat([all_test_predictions_df, current_fold_predictions], ignore_index=True)

        except Exception as e:
            print(f"Error during model training/evaluation for fold {fold_num + 1}: {e}")

    if not fold_results_list:
        print("No folds were successfully processed."); return

    print("\n\n--- K-FOLD MULTI-CLASS CV SUMMARY ---")
    cv_summary_df = pd.DataFrame(fold_results_list)
    
    print("Average Metrics Across All Folds:")
    avg_metrics = cv_summary_df.mean()
    print(f"Overall Accuracy: {avg_metrics.get('overall_accuracy', 0):.4f}")
    print(f"Macro Avg F1-Score: {avg_metrics.get('macro_avg_f1-score', 0):.4f}")
    print(f"Weighted Avg F1-Score: {avg_metrics.get('weighted_avg_f1-score', 0):.4f}")
    
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    summary_path = result_path.replace('.csv', '_summary.csv')
    details_path = result_path.replace('.csv', '_fold_details.csv')
    predictions_path = result_path.replace('.csv', '_all_predictions.csv')
    
    avg_metrics.to_frame('average_metrics').to_csv(summary_path)
    cv_summary_df.to_csv(details_path, index=False)
    all_test_predictions_df.to_csv(predictions_path, index=False)

    print(f"\nAggregated CV summary saved to: {summary_path}")
    print(f"Detailed fold results saved to: {details_path}")
    print(f"All test predictions saved to: {predictions_path}")

if __name__ == "__main__":
    matched_data_dir = r"C:\Users\misog\portfolio\Machine learning skin lesion project\matched_data" # Your provided path
    original_img_dir = os.path.join(matched_data_dir, "images")
    mask_img_dir = os.path.join(matched_data_dir, "masks")
    labels_csv_path = os.path.join(matched_data_dir, "metadata_matched.csv")

    output_dir = r'C:\Users\misog\portfolio\Machine learning skin lesion project\Skin-lesion-classification\result'
    os.makedirs(output_dir, exist_ok=True)
    merged_csv_filename = "dataset_multiclass_features.csv"
    output_csv_path = os.path.join(output_dir, merged_csv_filename)
    model_result_filename = "model_evaluation_multiclass.csv"
    result_path = os.path.join(output_dir, model_result_filename)

    if not os.path.exists(original_img_dir):
        print(f"ERROR: Matched data directory not found at '{original_img_dir}'.")
        sys.exit(1)

    try:
        # Run with recreate_features=True for the first time
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)