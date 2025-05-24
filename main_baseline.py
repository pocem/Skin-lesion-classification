--- START OF FILE main_baseline.py ---

# In main_baseline.py
import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
    from models_evaluation import train_and_select_model
except ImportError as e:
    print(f"Error: Could not import custom feature/model modules: {e}")
    print("Please ensure feature_A.py, feature_B.py, feature_C.py are in the 'util' directory (or adjust import paths).")
    print("Ensure models_evaluation.py is in the same directory or Python path.")
    print("The required functions are: extract_asymmetry_features, extract_border_features_from_folder, calculate_border_score, extract_feature_C, train_and_select_model.")
    sys.exit(1)

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None):
    print("Starting feature extraction process...")

    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")

    # --- Extract Asymmetry Features (Feature A) ---
    print(f"\nExtracting Asymmetry features from: {original_img_dir}")
    try:
        df_A = extract_asymmetry_features(folder_path=original_img_dir, output_csv=None, visualize=False)
        if df_A.empty:
            print("Warning: Asymmetry feature extraction (feature_A) returned an empty DataFrame.")
        else:
            print(f"Asymmetry features extracted: {df_A.shape[0]} images, {df_A.shape[1]-1} features (excluding filename).")
            if 'filename' not in df_A.columns and not df_A.empty:
                 print("CRITICAL WARNING: df_A is missing 'filename' column!")
    except Exception as e:
        print(f"Error during Asymmetry feature extraction: {e}")
        df_A = pd.DataFrame(columns=['filename']) 

    # --- Extract Border Features (Feature B) ---
    print(f"\nExtracting Border features from: {original_img_dir}")
    try:
        df_B_raw = extract_border_features_from_folder(folder_path=original_img_dir, output_csv=None, visualize=False)
        if df_B_raw.empty:
            print("Warning: Border feature extraction (feature_B raw) returned an empty DataFrame.")
            df_B = pd.DataFrame(columns=['filename'])
        else:
            print(f"Raw Border features extracted: {df_B_raw.shape[0]} images, {df_B_raw.shape[1]-1} features.")
            if 'filename' not in df_B_raw.columns and not df_B_raw.empty:
                print("CRITICAL WARNING: df_B_raw is missing 'filename' column!")
            df_B = calculate_border_score(df_B_raw)
            print(f"Border scores calculated. Total border features df: {df_B.shape[0]} images, {df_B.shape[1]-1} features.")
            cols_to_drop_from_B = ['sobel_mean_safe', 'avg_contour_perimeter_safe', 'laplacian_mean_safe', 'avg_contour_area_safe']
            df_B = df_B.drop(columns=[col for col in cols_to_drop_from_B if col in df_B.columns], errors='ignore')
    except Exception as e:
        print(f"Error during Border feature extraction: {e}")
        df_B = pd.DataFrame(columns=['filename'])

    # --- Extract Color Features (Feature C) ---
    print(f"\nExtracting Color features from: {original_img_dir}")
    try:
        df_C = extract_feature_C(folder_path=original_img_dir, output_csv=None, normalize_colors=True, visualize=False)
        if df_C.empty:
            print("Warning: Color feature extraction (feature_C) returned an empty DataFrame.")
        else:
            print(f"Color features extracted: {df_C.shape[0]} images, {df_C.shape[1]-1} features.")
            if 'filename' not in df_C.columns and not df_C.empty:
                 print("CRITICAL WARNING: df_C is missing 'filename' column!")
    except Exception as e:
        print(f"Error during Color feature extraction: {e}")
        df_C = pd.DataFrame(columns=['filename'])

    # --- Load Labels / Metadata ---
    metadata_df = None
    if labels_csv and exists(labels_csv):
        print(f"\nLoading metadata from {labels_csv}")
        try:
            raw_metadata_df = pd.read_csv(labels_csv)
            if 'img_id' in raw_metadata_df.columns:
                raw_metadata_df = raw_metadata_df.rename(columns={'img_id': 'filename'})
            
            if 'filename' not in raw_metadata_df.columns:
                print("Warning: Metadata CSV must contain 'filename' (or 'img_id') column. Proceeding without metadata.")
            elif 'diagnostic' not in raw_metadata_df.columns:
                print("Warning: Metadata CSV must contain 'diagnostic' column for model training. Proceeding without labels.")
            else:
                # Keep only filename and diagnostic for merging, other metadata can be added if needed for features later
                metadata_df = raw_metadata_df[['filename', 'diagnostic']] # Add other columns if they are features
                print(f"Metadata (filename, diagnostic) loaded: {metadata_df.shape[0]} entries.")
        except Exception as e:
            print(f"Error loading metadata: {e}. Proceeding without metadata.")
    else:
        print("\nNo metadata file provided or file doesn't exist. Proceeding without metadata.")

    # --- Merge DataFrames ---
    print("\nMerging feature DataFrames...")
    
    dataframes_to_merge = []
    if metadata_df is not None and not metadata_df.empty and 'filename' in metadata_df.columns:
        dataframes_to_merge.append(metadata_df)
    
    # Add feature DataFrames only if they are not empty and have 'filename'
    if not df_A.empty and 'filename' in df_A.columns: dataframes_to_merge.append(df_A)
    elif not df_A.empty: print("Skipping df_A in merge due to missing 'filename' column or being empty.")
        
    if not df_B.empty and 'filename' in df_B.columns: dataframes_to_merge.append(df_B)
    elif not df_B.empty: print("Skipping df_B in merge due to missing 'filename' column or being empty.")

    if not df_C.empty and 'filename' in df_C.columns: dataframes_to_merge.append(df_C)
    elif not df_C.empty: print("Skipping df_C in merge due to missing 'filename' column or being empty.")

    if not dataframes_to_merge:
        print("No DataFrames with 'filename' column to merge. Exiting feature creation.")
        return pd.DataFrame()
    
    if len(dataframes_to_merge) == 1 and metadata_df is not None and dataframes_to_merge[0] is metadata_df:
        print("Only metadata DataFrame is available. No features to merge. Saving metadata.")
        final_df = metadata_df
    elif len(dataframes_to_merge) < 2 : # Needs at least one feature df + optionally metadata, or two feature dfs
        print("Not enough DataFrames to perform a meaningful merge (need at least one feature set).")
        if dataframes_to_merge: # if one feature df exists
            final_df = dataframes_to_merge[0]
        else: # if only metadata_df or nothing
            return pd.DataFrame()
    else:
        final_df = dataframes_to_merge[0]
        for df_to_merge in dataframes_to_merge[1:]:
            final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner') # 'inner' to keep only rows with all features and metadata

    if final_df.empty:
        print("Resulting merged DataFrame is empty. This might be due to 'inner' merge and no common filenames or issues with feature extraction.")
    else:
        print(f"Merged DataFrame shape: {final_df.shape}")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")
    
    if not final_df.empty:
        feature_cols = [col for col in final_df.columns if col not in ['filename', 'diagnostic']] # 'label' will be created later
        print(f"Dataset contains {len(feature_cols)} potential feature columns (excluding filename/diagnostic).")
    
    return final_df

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- FEATURE DATASET CREATION ---\n")
    
    if not original_img_dir or not output_csv_path:
        raise ValueError("original_img_dir and output_csv_path must be provided.")

    data_df = None
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        try:
            data_df = pd.read_csv(output_csv_path)
        except Exception as e:
            print(f"Error loading existing dataset: {e}. Will attempt to recreate.")
            data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)

    if data_df is None or data_df.empty:
        print("Failed to create or load the feature dataset. Exiting.")
        return

    print("\n--- Merged Dataset Information ---")
    data_df.info()
    print("\nFirst 5 rows of the merged dataset:")
    print(data_df.head())
    
    if 'filename' not in data_df.columns:
        print("CRITICAL ERROR: 'filename' column is missing in the final DataFrame!")
        return
    if data_df['filename'].isnull().any():
        print("Warning: Some 'filename' entries are NaN. This usually indicates an issue in merging.")
    if data_df['filename'].duplicated().any():
        print("Warning: Duplicate filenames found. Consolidating by keeping the first occurrence.")
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first')


    print("\n--- MODEL TRAINING AND EVALUATION ---")
    
    # 1. Label Preparation
    if 'diagnostic' not in data_df.columns:
        print("CRITICAL ERROR: 'diagnostic' column not found in the dataset. Cannot proceed with model training.")
        return
    
    data_df.dropna(subset=['diagnostic'], inplace=True) # Drop rows where diagnostic label is missing
    if data_df.empty:
        print("CRITICAL ERROR: Dataset became empty after dropping rows with missing 'diagnostic' labels.")
        return

    le = LabelEncoder()
    data_df['label'] = le.fit_transform(data_df['diagnostic'])
    class_names = le.classes_
    print("\nDiagnostic classes and their encoded labels:")
    for i, class_name_item in enumerate(class_names):
        print(f"{class_name_item}: {i}")

    # 2. Feature Engineering (One-Hot Encoding for categorical features)
    if 'c_dominant_channel' in data_df.columns:
        print("\nOne-hot encoding 'c_dominant_channel'...")
        data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False) # dummy_na=False to not create NaN column
    
    # 3. Identify Feature Columns
    # Exclude identifiers, original diagnostic, and the new numerical label
    potential_non_feature_cols = ['filename', 'diagnostic', 'label', 
                                  'patient_id', 'lesion_id', 'smoke', 'drink', 
                                  'background_father', 'background_mother', 'age', 'pesticide', 
                                  'gender', 'skin_cancer_history', 'cancer_history', 'has_piped_water', 
                                  'has_sewage_system', 'fitspatrick', 'region', 'diameter_1', 
                                  'diameter_2', 'itch', 'grew', 'hurt', 'changed', 'bleed', 
                                  'elevation', 'biopsed'] # Add any other known non-feature columns from metadata
    
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]
    
    if not feature_columns:
        print("CRITICAL ERROR: No feature columns identified after exclusions. Cannot train model.")
        return
    print(f"\nUsing {len(feature_columns)} feature columns for training: {feature_columns}")

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy()
    current_filenames = data_df['filename'].copy()


    # 4. Data Cleaning (Convert to numeric, Impute NaNs and Infs)
    print("\nConverting features to numeric and handling NaNs/Infs...")
    for feat in feature_columns:
        x_all[feat] = pd.to_numeric(x_all[feat], errors='coerce')

    all_nan_cols = x_all.columns[x_all.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Warning: The following columns became all NaN after numeric conversion and will be dropped: {all_nan_cols}")
        x_all = x_all.drop(columns=all_nan_cols)
        feature_columns = [col for col in feature_columns if col not in all_nan_cols]
        if not feature_columns:
            print("CRITICAL ERROR: All feature columns were dropped. Cannot train model.")
            return
    
    # Replace Inf with NaN before imputation
    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy='mean')
    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index)

    # Ensure y_all and current_filenames are aligned with x_all if any rows were dropped due to all-NaN features (unlikely here)
    # This alignment is generally robust due to pandas indexing.

    if len(x_all) == 0:
        print("Skipping model training: No samples remaining after data cleaning.")
        return
    if y_all.nunique() < 2:
        print("Skipping model training: Not enough unique classes in labels for stratified split or training.")
        return

    # 5. Data Splitting (60% train, 20% validation, 20% test)
    print(f"\nSplitting data into train, validation, and test sets (Total samples: {len(x_all)})...")
    try:
        x_train, x_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
            x_all, y_all, current_filenames, test_size=0.4, random_state=42, stratify=y_all
        )
        x_val, x_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
            x_temp, y_temp, filenames_temp, test_size=0.5, random_state=42, stratify=y_temp # 0.5 of 0.4 is 0.2
        )
    except ValueError as e_split:
        print(f"Error during data splitting: {e_split}. This might be due to too few samples in some classes.")
        print(f"Class distribution in y_all: \n{y_all.value_counts()}")
        return

    print(f"Training set size: {len(x_train)}")
    print(f"Validation set size: {len(x_val)}")
    print(f"Test set size: {len(x_test)}")

    # 6. Model Training and Selection
    try:
        best_model, best_model_name, best_val_acc = train_and_select_model(x_train, y_train, x_val, y_val)

        # 7. Test Phase
        print("\n--- TEST PHASE ---")
        y_test_pred = best_model.predict(x_test)
        
        # Check if predict_proba is available (not for all models like some SVMs by default)
        if hasattr(best_model, "predict_proba"):
            y_test_pred_proba = best_model.predict_proba(x_test)
        else:
            # Create a placeholder if predict_proba is not available
            # This will have uniform probabilities if num_classes > 1, or [0,1]/[1,0] for binary based on prediction
            print(f"Warning: Model {best_model_name} does not have predict_proba. Probabilities will be estimated.")
            y_test_pred_proba = np.zeros((len(y_test_pred), len(class_names)))
            for i, pred_label in enumerate(y_test_pred):
                y_test_pred_proba[i, pred_label] = 1.0


        test_acc = accuracy_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)
        # Ensure labels for confusion matrix match the order of class_names if some classes are not in y_test/y_test_pred
        cm_labels = np.arange(len(class_names)) 
        cm_display = confusion_matrix(y_test, y_test_pred, labels=cm_labels)


        cls_report_dict = classification_report(y_test, y_test_pred, labels=cm_labels, target_names=class_names, output_dict=True, zero_division=0)
        cls_report_str = classification_report(y_test, y_test_pred, labels=cm_labels, target_names=class_names, zero_division=0)
        
        print(f"\nBest Model on Test Set: {best_model_name}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix (Test Set):\n{cm_display}")
        print(f"Classes for CM display: {class_names}")
        print(f"Classification Report (Test Set):\n{cls_report_str}")

        # 8. Reporting
        # Detailed Test Results CSV
        test_results_df = pd.DataFrame({
            'filename': filenames_test.values, # Ensure it's an array/list
            'true_label_encoded': y_test.values,
            'predicted_label_encoded': y_test_pred,
            'true_label_diagnostic': le.inverse_transform(y_test.values),
            'predicted_label_diagnostic': le.inverse_transform(y_test_pred)
        })
        for i, class_name_item in enumerate(class_names):
            if i < y_test_pred_proba.shape[1]:
                 test_results_df[f'proba_{class_name_item}'] = y_test_pred_proba[:, i]
            else: # Should not happen if y_test_pred_proba is correctly sized
                 test_results_df[f'proba_{class_name_item}'] = 0.0 
        
        # Ensure result directory exists for reports
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        test_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_predictions_details.csv")
        test_results_df.to_csv(test_details_csv_path, index=False)
        print(f"Detailed test predictions saved to {test_details_csv_path}")

        # Summary Report CSV
        summary_report_data = {
            'model_name': best_model_name,
            'validation_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'num_training_samples': len(x_train),
            'num_validation_samples': len(x_val),
            'num_test_samples': len(x_test),
            'num_features_used': len(feature_columns),
            # 'feature_columns_list': ", ".join(feature_columns) # Can be very long
        }
        for class_label_report in class_names:
            if class_label_report in cls_report_dict:
                summary_report_data[f'{class_label_report}_precision_test'] = cls_report_dict[class_label_report]['precision']
                summary_report_data[f'{class_label_report}_recall_test'] = cls_report_dict[class_label_report]['recall']
                summary_report_data[f'{class_label_report}_f1-score_test'] = cls_report_dict[class_label_report]['f1-score']
                summary_report_data[f'{class_label_report}_support_test'] = cls_report_dict[class_label_report]['support']
        
        if 'macro avg' in cls_report_dict:
            summary_report_data['macro_avg_precision_test'] = cls_report_dict['macro avg']['precision']
            summary_report_data['macro_avg_recall_test'] = cls_report_dict['macro avg']['recall']
            summary_report_data['macro_avg_f1-score_test'] = cls_report_dict['macro avg']['f1-score']
        if 'weighted avg' in cls_report_dict:
            summary_report_data['weighted_avg_precision_test'] = cls_report_dict['weighted avg']['precision']
            summary_report_data['weighted_avg_recall_test'] = cls_report_dict['weighted avg']['recall']
            summary_report_data['weighted_avg_f1-score_test'] = cls_report_dict['weighted avg']['f1-score']

        results_summary_df = pd.DataFrame([summary_report_data])
        results_summary_df.to_csv(result_path, index=False)
        print(f"Summary model evaluation results saved to {result_path}")

    except ImportError:
        print("Skipping model training: 'models_evaluation' module or 'train_and_select_model' function not found.")
    except Exception as e_model:
        print(f"Error during model training/evaluation: {e_model}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure paths - ADJUST THESE TO YOUR SPECIFIC FOLDERS
    # Assuming 'util' folder is at the same level as main_baseline.py or in Python path
    # Using the more specific paths from your second __main__ block:
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks"
    labels_csv_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"
    
    # Define output path for the final merged CSV (features + metadata for training)
    merged_csv_filename = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\result\dataset_baseline_features_with_metadata.csv" # More descriptive name
    # Place it in a 'data/processed' directory to keep things organized might be good, for now use result
    output_feature_csv_dir = "./result"
    os.makedirs(output_feature_csv_dir, exist_ok=True)
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)
    
    # Result path for model evaluation summary
    model_result_filename = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\result\model_evaluation_csv"  # More descriptive
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)

    # Ensure result directory exists
    os.makedirs(output_feature_csv_dir, exist_ok=True) # Redundant if using output_feature_csv_dir above
    
    try:
        # Set recreate_features=True to always regenerate the feature CSV
        # Set to False to load if exists, or create if not.
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)