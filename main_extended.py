

# In main_extended.py
import sys
import os 
from os.path import join, exists
import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shutil # Ensure shutil is imported here

# Import custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
    from util.contrast_feature import extract_feature_contrast # USE CONTRAST
    from util.blue_veil import extract_feature_BV
    from util.hair_removal_feature import remove_and_save_hairs 
    from models_evaluation import train_and_select_model
except ImportError as e:
    print(f"Error: Could not import custom feature/model modules: {e}")
    print("Please ensure all feature modules (feature_A.py, feature_B.py, feature_C.py, contrast_feature.py, blue_veil.py, hair_removal_feature.py) are in the 'util' directory.")
    print("Ensure models_evaluation.py is in the same directory or Python path.")
    sys.exit(1)

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None, recreate_features=False):
    print("Starting EXTENDED feature extraction process (with Contrast)...")

    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")

    base_output_dir = os.path.dirname(output_csv_path)
    # The hair_removed_img_dir_path is indeed inside result_extended_contrast
    hair_removed_img_dir_path = os.path.join(base_output_dir, "hair_removed_images_extended_contrast") 
    if recreate_features:
        print(f"Recreate features is True, removing existing hair-removed images directory: {hair_removed_img_dir_path}")
        if exists(hair_removed_img_dir_path):
            shutil.rmtree(hair_removed_img_dir_path)
        else:
            print(f"Hair-removed images directory '{hair_removed_img_dir_path}' does not exist, creating new.")
    
    os.makedirs(hair_removed_img_dir_path, exist_ok=True)
    
    print(f"\nHair removal processing: Original images from '{original_img_dir}'")
    print(f"Processed images will be stored in: '{hair_removed_img_dir_path}'")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    try:
        original_image_files = [f for f in os.listdir(original_img_dir) if f.lower().endswith(valid_extensions)]
        print(f"Found {len(original_image_files)} images in original directory for hair processing.")
    except Exception as e:
        print(f"Error listing files in original_img_dir '{original_img_dir}': {e}")
        return pd.DataFrame()

    hair_params = { 
        "blackhat_kernel_size": (15, 15), "threshold_value": 18,
        "dilation_kernel_size": (3, 3), "dilation_iterations": 2,
        "inpaint_radius": 5, "min_hair_contours_to_process": 3,
        "min_contour_area": 15
    }
    
    processed_files_in_loop = 0
    inpainted_count = 0
    copied_original_count = 0
    hair_removal_error_count = 0
    
    # NEW: List to store hair counts for each image
    hair_counts_data = []

    for filename in tqdm(original_image_files, desc="Performing Hair Removal"):
        original_image_path = os.path.join(original_img_dir, filename)
        target_path_in_hair_removed_dir = os.path.join(hair_removed_img_dir_path, filename)

        # Initialize current hair count for this file
        current_hair_count = 0 

        # This skip logic is applied: if not recreating features AND the file already exists,
        # we assume its hair count was captured in a previous run and assign 0 for *this* run's processing
        if not recreate_features and os.path.exists(target_path_in_hair_removed_dir):
            hair_counts_data.append({'filename': filename, 'num_hairs': current_hair_count})
            processed_files_in_loop +=1
            continue # Skip to next file
        
        try:
            # Call hair removal function, which returns the count of hairs found
            hair_count_for_file, saved_img_path, msg = remove_and_save_hairs(
                image_path=original_image_path,
                output_dir=hair_removed_img_dir_path,
                **hair_params
            )
            current_hair_count = hair_count_for_file # Capture the returned hair count
            
            if "hairs removed" in msg and saved_img_path and os.path.exists(saved_img_path):
                inpainted_count += 1
            elif ("No significant hairs found" in msg or "original image skipped" in msg) and \
                 not os.path.exists(target_path_in_hair_removed_dir) :
                shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
            elif not os.path.exists(target_path_in_hair_removed_dir): 
                print(f"\nWarning: Hair removal for {filename} - message: '{msg}'. Output file not found. Copying original.")
                shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
            
            processed_files_in_loop +=1

        except FileNotFoundError:
             print(f"\nError: Original image {filename} not found at {original_image_path} during hair removal loop.")
             hair_removal_error_count +=1
             # current_hair_count remains 0
        except Exception as e_hair:
            print(f"\nError during hair removal for {filename}: {e_hair}. Attempting to copy original.")
            hair_removal_error_count += 1
            # current_hair_count remains 0
            try:
                if not os.path.exists(target_path_in_hair_removed_dir):
                    shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
                processed_files_in_loop += 1
            except Exception as e_copy:
                print(f"\nFailed to copy original {filename} after hair removal error: {e_copy}")
        
        # Ensure hair count is recorded for every file, even if errors or skips occurred
        hair_counts_data.append({'filename': filename, 'num_hairs': current_hair_count})

    print(f"\nHair removal loop summary: Files processed/checked in loop: {processed_files_in_loop}, Actually inpainted: {inpainted_count}, Originals copied: {copied_original_count}, Errors during processing: {hair_removal_error_count}")
    
    # NEW: Create DataFrame for hair counts
    df_hair_counts = pd.DataFrame(hair_counts_data)
    print(f"Hair count features extracted. Shape: {df_hair_counts.shape}")

    feature_processing_dir = hair_removed_img_dir_path
    num_images_for_features = len(os.listdir(feature_processing_dir))
    if num_images_for_features == 0:
        print(f"CRITICAL: No images found in '{feature_processing_dir}' for feature extraction. Exiting.")
        return pd.DataFrame()
    print(f"Total images in '{feature_processing_dir}' for subsequent feature extraction: {num_images_for_features}")

    dfs = {}
    feature_extractors = {
        "A": extract_asymmetry_features,
        "B_raw": extract_border_features_from_folder, 
        "C": extract_feature_C,
        "Contrast": extract_feature_contrast, # USING CONTRAST
        "BV": extract_feature_BV
    }

    # NEW: Add hair counts DataFrame to the collection of DataFrames to be merged
    dfs["Hair_Counts"] = df_hair_counts 

    for name, func in feature_extractors.items():
        print(f"\nExtracting {name} features from: {feature_processing_dir}")
        try:
            if name == "B_raw":
                df_temp_raw = func(folder_path=feature_processing_dir, output_csv=None, visualize=False)
                if not df_temp_raw.empty:
                    dfs["B"] = calculate_border_score(df_temp_raw)
                    cols_to_drop_from_B = ['sobel_mean_safe', 'avg_contour_perimeter_safe', 'laplacian_mean_safe', 'avg_contour_area_safe']
                    dfs["B"] = dfs["B"].drop(columns=[col for col in cols_to_drop_from_B if col in dfs["B"].columns], errors='ignore')
                    print(f"Border features (B) processed. Shape: {dfs['B'].shape}")
                else:
                    print(f"Warning: {name} feature extraction returned an empty DataFrame.")
                    dfs["B"] = pd.DataFrame(columns=['filename'])
            else:
                dfs[name] = func(folder_path=feature_processing_dir, output_csv=None, visualize=False)
                if dfs[name].empty:
                    print(f"Warning: {name} feature extraction returned an empty DataFrame.")
                else:
                    print(f"{name} features extracted. Shape: {dfs[name].shape}")
        except Exception as e:
            print(f"Error during {name} feature extraction: {e}")
            dfs[name] = pd.DataFrame(columns=['filename'])

    metadata_df = None
    if labels_csv and exists(labels_csv):
        print(f"\nLoading metadata from {labels_csv}")
        try:
            raw_metadata_df = pd.read_csv(labels_csv)
            if 'img_id' in raw_metadata_df.columns:
                raw_metadata_df = raw_metadata_df.rename(columns={'img_id': 'filename'})
            label_column_name = 'diagnostic'
            if 'filename' not in raw_metadata_df.columns or label_column_name not in raw_metadata_df.columns:
                print(f"ERROR: Metadata CSV must contain 'filename' (or 'img_id') and '{label_column_name}'.")
            else:
                if label_column_name != 'real_label':
                    raw_metadata_df.rename(columns={label_column_name: 'real_label'}, inplace=True)
                cancer_diagnoses = ["BCC", "SCC", "MEL"]
                raw_metadata_df['binary_target'] = raw_metadata_df['real_label'].apply(lambda x: 1 if x in cancer_diagnoses else 0)
                cols_to_keep_from_metadata = ['filename', 'real_label', 'binary_target']
                metadata_df = raw_metadata_df[cols_to_keep_from_metadata].copy()
                print(f"Metadata (filename, real_label, binary_target) selected. Shape: {metadata_df.shape[0]} entries.")
        except Exception as e: print(f"Error loading metadata: {e}")
    else: print("\nNo metadata file provided or found. Proceeding without metadata.")

    print("\nMerging feature DataFrames...")
    dataframes_to_merge = []
    if metadata_df is not None and not metadata_df.empty:
        dataframes_to_merge.append(metadata_df)
    
    for name, df in dfs.items():
        if name == "B_raw": continue
        if not df.empty and 'filename' in df.columns:
            dataframes_to_merge.append(df)
            print(f"df_{name} added for merge. Shape: {df.shape}")
        else:
             print(f"df_{name} is empty or missing 'filename', not added to merge.")

    if not dataframes_to_merge or (metadata_df is None and len(dataframes_to_merge) < 1) or \
       (metadata_df is not None and len(dataframes_to_merge) < 2): 
        print("Not enough Dataframes to merge meaningfully. Exiting feature creation.")
        return pd.DataFrame()
            
    final_df = dataframes_to_merge[0]
    print(f"Base DataFrame for merge: Columns: {final_df.columns.tolist()[:5]}..., Shape: {final_df.shape}")

    for i, df_to_merge in enumerate(dataframes_to_merge[1:]):
        df_name_for_log = "Unknown_DF"
        for key_name, val_df in dfs.items():
            if val_df is df_to_merge: df_name_for_log = key_name; break
        
        print(f"Merging with DataFrame for feature '{df_name_for_log}' (shape {df_to_merge.shape})")
        common_filenames = set(final_df['filename']).intersection(set(df_to_merge['filename']))
        if not common_filenames:
            print(f"CRITICAL WARNING: No common 'filename' with {df_name_for_log}. Skipping merge."); continue 
        final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner')
        if final_df.empty: print(f"CRITICAL WARNING: DataFrame empty after merging with {df_name_for_log}."); break 
        else: print(f"Shape after merging with {df_name_for_log}: {final_df.shape}")
            
    if final_df.empty: print("Resulting merged DataFrame is empty.")
    else: print(f"Merged DataFrame final shape: {final_df.shape}. Columns: {final_df.columns.tolist()}")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")
    
    if not final_df.empty:
        expected_label_cols = ['real_label', 'binary_target']
        missing = [col for col in expected_label_cols if col not in final_df.columns]
        if missing: print(f"WARNING: Final DataFrame MISSING: {missing}")
        else: print(f"SUCCESS: 'real_label' and 'binary_target' columns present.")
    return final_df

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- FEATURE DATASET CREATION (EXTENDED FEATURES - Contrast) ---\n")
    
    if not original_img_dir or not output_csv_path:
        raise ValueError("original_img_dir and output_csv_path must be provided.")

    data_df = None
    # Always force recreation of the entire dataset for consistency with hair counting.
    # This ensures that the hair-removed images are always fresh.
    print(f"Force recreating entire dataset for consistent hair count extraction.")
    # If the output CSV exists, delete it to ensure a fresh start.
    if exists(output_csv_path):
        print(f"Deleting existing output CSV: {output_csv_path}")
        os.remove(output_csv_path)

    data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, 
                                     labels_csv=labels_csv_path, recreate_features=True) # Always True here
    
    if data_df is None or data_df.empty:
        print("Failed to create or load the feature dataset. Exiting.")
        return
    
    print("\n--- Merged Dataset Information ---")
    data_df.info(verbose=True, show_counts=True) 
    print("\nFirst 5 rows of the merged dataset:")
    print(data_df.head())
    
    if 'filename' not in data_df.columns:
        print("CRITICAL ERROR: 'filename' column is missing in the final DataFrame!")
        return
    if data_df['filename'].isnull().any():
        print(f"Warning: {data_df['filename'].isnull().sum()} 'filename' entries are NaN.")
    if data_df['filename'].duplicated().any():
        print(f"Warning: {data_df['filename'].duplicated().sum()} duplicate filenames found. Consolidating.")
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)

    print("\n--- MODEL TRAINING AND EVALUATION ---")
    
    if 'binary_target' not in data_df.columns:
        source_label_col = next((col for col in ['real_label', 'diagnostic'] if col in data_df.columns), None)
        if source_label_col:
            print(f"Creating 'binary_target' from '{source_label_col}'.")
            cancer_diagnoses_map = ["BCC", "SCC", "MEL"]
            data_df['binary_target'] = data_df[source_label_col].apply(lambda x: 1 if x in cancer_diagnoses_map else 0)
            if source_label_col == 'diagnostic' and 'real_label' not in data_df.columns:
                data_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
        else:
            print("CRITICAL ERROR: Target label column not found."); return
    
    data_df.dropna(subset=['binary_target'], inplace=True) 
    if data_df.empty: print("CRITICAL ERROR: Dataset empty after dropping NaNs in 'binary_target'."); return

    data_df['label'] = data_df['binary_target'].astype(int)
    class_names_for_report = ['non-cancer', 'cancer'] 
    print(f"\nLabel distribution:\n{data_df['label'].value_counts(normalize=True)}")

    if 'real_label' not in data_df.columns: # Ensure 'real_label' exists for reporting
        data_df['real_label'] = data_df['label'].map({0: 'derived_non-cancer', 1: 'derived_cancer'})

    if 'c_dominant_channel' in data_df.columns:
        print("\nOne-hot encoding 'c_dominant_channel'...")
        try: data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False)
        except Exception as e_ohe: print(f"Error OHE 'c_dominant_channel': {e_ohe}")
    
    potential_non_feature_cols = ['filename', 'real_label', 'binary_target', 'label', 'diagnostic', 
                                  'patient_id', 'lesion_id', 'smoke', 'drink', 'background_father', 
                                  'background_mother', 'age', 'pesticide', 'gender', 
                                  'skin_cancer_history', 'cancer_history', 'has_piped_water', 
                                  'has_sewage_system', 'fitspatrick', 'region', 'diameter_1', 
                                  'diameter_2', 'itch', 'grew', 'hurt', 'changed', 'bleed', 
                                  'elevation', 'biopsed'] 
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]
    
    if not feature_columns: print("CRITICAL ERROR: No feature columns identified."); return
    print(f"\nUsing {len(feature_columns)} feature columns. First 10: {feature_columns[:10]}...")

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy() 
    current_filenames = data_df['filename'].copy()

    print("\nConverting features to numeric, handling NaNs/Infs...")
    for feat in feature_columns: x_all[feat] = pd.to_numeric(x_all[feat], errors='coerce')

    all_nan_cols = x_all.columns[x_all.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Columns dropped due to all NaN: {all_nan_cols}")
        x_all = x_all.drop(columns=all_nan_cols)
        feature_columns = [col for col in feature_columns if col not in all_nan_cols]
        if not feature_columns: print("CRITICAL ERROR: All feature columns dropped."); return
    
    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean') 
    if x_all.empty: print("CRITICAL ERROR: x_all empty before imputation."); return
    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index)

    if len(x_all) == 0 or y_all.nunique() < 2:
        print(f"Skipping model training: Samples: {len(x_all)}, Unique Labels: {y_all.nunique()}"); return

    print(f"\nSplitting data (Total samples: {len(x_all)})...")
    try:
        x_train, x_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
            x_all.reset_index(drop=True), y_all.reset_index(drop=True), current_filenames.reset_index(drop=True), 
            test_size=0.4, random_state=42, stratify=y_all.reset_index(drop=True)
        )
        x_val, x_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
            x_temp.reset_index(drop=True), y_temp.reset_index(drop=True), filenames_temp.reset_index(drop=True), 
            test_size=0.5, random_state=42, stratify=y_temp.reset_index(drop=True)
        )
    except ValueError as e_split: print(f"Error data splitting: {e_split}. Labels: \n{y_all.value_counts()}"); return

    print(f"Training set: {len(x_train)}, Validation set: {len(x_val)}, Test set: {len(x_test)}")

    try:
        if x_train.empty or x_val.empty or y_train.nunique() < 2:
            print("Training/validation empty or insufficient classes. Skipping model training."); return

        best_model, best_model_name, best_val_acc = train_and_select_model(x_train, y_train, x_val, y_val)
        if best_model is None: print("No model selected. Exiting."); return

        print("\n--- TEST PHASE ---")
        y_test_pred = best_model.predict(x_test)
        y_test_pred_proba = None
        if hasattr(best_model, "predict_proba"): y_test_pred_proba = best_model.predict_proba(x_test)
        else:
            print(f"Warning: Model {best_model_name} no predict_proba.")
            y_test_pred_proba = np.zeros((len(y_test_pred), len(class_names_for_report)))
            for i, pred_label in enumerate(y_test_pred): y_test_pred_proba[i, pred_label] = 1.0

        test_acc = accuracy_score(y_test, y_test_pred)
        cm_display = confusion_matrix(y_test, y_test_pred, labels=[0,1])
        cls_report_dict = classification_report(y_test, y_test_pred, labels=[0,1], target_names=class_names_for_report, output_dict=True, zero_division=0)
        cls_report_str = classification_report(y_test, y_test_pred, labels=[0,1], target_names=class_names_for_report, zero_division=0)
        
        print(f"\nBest Model: {best_model_name}, Test Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{cm_display}\nClassification Report:\n{cls_report_str}")

        test_results_df = pd.DataFrame({
            'filename': filenames_test.reset_index(drop=True).values,
            'true_label_encoded': y_test.reset_index(drop=True).values,
            'predicted_label_encoded': y_test_pred,
            'true_label_text': y_test.reset_index(drop=True).map({0: 'non-cancer', 1: 'cancer'}).values,
            'predicted_label_text': pd.Series(y_test_pred).map({0: 'non-cancer', 1: 'cancer'}).values
        })
        if y_test_pred_proba is not None and y_test_pred_proba.shape[0] == len(y_test_pred) and y_test_pred_proba.shape[1] >= len(class_names_for_report):
            test_results_df[f'proba_{class_names_for_report[0]}'] = y_test_pred_proba[:, 0]
            test_results_df[f'proba_{class_names_for_report[1]}'] = y_test_pred_proba[:, 1]
        
        test_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_predictions_details.csv")
        test_results_df.to_csv(test_details_csv_path, index=False)
        print(f"Detailed test predictions saved to {test_details_csv_path}")

        summary_report_data = {
            'model_name': best_model_name, 'validation_accuracy': best_val_acc, 'test_accuracy': test_acc,
            'num_training_samples': len(x_train), 'num_validation_samples': len(x_val), 'num_test_samples': len(x_test),
            'num_features_used': len(feature_columns),
        }
        for cl_label in class_names_for_report: 
            if cl_label in cls_report_dict:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    summary_report_data[f'{cl_label}_{metric}_test'] = cls_report_dict[cl_label][metric]
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in cls_report_dict:
                 for metric in ['precision', 'recall', 'f1-score']:
                    summary_report_data[f'{avg_type.replace(" ", "_")}_{metric}_test'] = cls_report_dict[avg_type][metric]

        results_summary_df = pd.DataFrame([summary_report_data])
        results_summary_df.to_csv(result_path, index=False)
        print(f"Summary model evaluation results saved to {result_path}")

    except Exception as e_model:
        print(f"Error during model training/evaluation: {e_model}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks" 
    labels_csv_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"
    
    output_feature_csv_dir = "./result_extended_contrast" 
    os.makedirs(output_feature_csv_dir, exist_ok=True)
    
    # These filenames will be overwritten on each run due to recreate_features=True
    # DO NOT CHANGE THESE NAMES FOR SUBSEQUENT RUNS IF YOU WANT OVERWRITING
    merged_csv_filename = "dataset_extended_features_ABC_Contrast_BV_Hairs_Definitive.csv" 
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)
    
    model_result_filename = "model_evaluation_extended_contrast_hairs_Definitive_summary.csv" 
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)
    
    try:
        # Set recreate_features=True to force regeneration of hair-removed images and feature CSV.
        # This is CRUCIAL to get accurate hair counts if the processed images exist from a prior run.
        # It also ensures the output CSVs (defined above) are overwritten.
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True) 
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
