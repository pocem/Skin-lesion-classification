# In main_baseline.py
import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shutil # For copying files

# Import custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
    # Assuming the function in haralick_extended.py is indeed named extract_haralick_features
    # If it's extract_feature_H, change the import or use an alias:
    # from util.haralick_extended import extract_feature_H as extract_haralick_features
    from util.haralick_extended import extract_haralick_features
    from util.blue_veil import extract_feature_BV
    from util.hair_removal_feature import remove_and_save_hairs # Import the specific function
    from models_evaluation import train_and_select_model
except ImportError as e:
    print(f"Error: Could not import custom feature/model modules: {e}")
    print("Please ensure all feature modules (feature_A.py, feature_B.py, feature_C.py, haralick_extended.py, blue_veil.py, hair_removal_feature.py) are in the 'util' directory (or adjust import paths).")
    print("Ensure models_evaluation.py is in the same directory or Python path.")
    sys.exit(1)

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None, recreate_features=False):
    print("Starting feature extraction process...")

    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")

    # --- HAIR REMOVAL STEP ---
    # Define a directory for hair-removed images, relative to the output CSV's directory
    hair_removed_img_dir_path = os.path.join(os.path.dirname(output_csv_path), "hair_removed_images")
    
    if recreate_features and exists(hair_removed_img_dir_path):
        print(f"Recreate features is True, removing existing hair-removed images directory: {hair_removed_img_dir_path}")
        shutil.rmtree(hair_removed_img_dir_path)
    os.makedirs(hair_removed_img_dir_path, exist_ok=True)
    
    print(f"\nHair removal process: Original images from '{original_img_dir}'")
    print(f"Processed images (hair-removed or original) will be in: '{hair_removed_img_dir_path}'")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    hair_params = { # Default parameters for hair removal, can be tuned
        "blackhat_kernel_size": (15, 15), "threshold_value": 18,
        "dilation_kernel_size": (3, 3), "dilation_iterations": 2,
        "inpaint_radius": 5, "min_hair_contours_to_process": 3,
        "min_contour_area": 15
    }
    processed_hair_removal_count = 0
    copied_original_count = 0
    hair_removal_error_count = 0

    original_image_files = [f for f in os.listdir(original_img_dir) if f.lower().endswith(valid_extensions)]
    
    for filename in original_image_files:
        original_image_path = os.path.join(original_img_dir, filename)
        target_path_in_hair_removed_dir = os.path.join(hair_removed_img_dir_path, filename)

        # If not recreating features and file exists, skip hair removal for this file
        if not recreate_features and os.path.exists(target_path_in_hair_removed_dir):
            # print(f"Skipping hair removal for {filename}, already exists in target and not recreating features.")
            processed_hair_removal_count +=1 # Count as processed for this stage
            continue
            
        try:
            hair_count, saved_img_path, msg = remove_and_save_hairs(
                image_path=original_image_path,
                output_dir=hair_removed_img_dir_path, # remove_and_save_hairs saves here if hairs are processed
                **hair_params
            )
            
            # The remove_and_save_hairs function saves the inpainted image if hair_count >= min_hair_contours_to_process.
            # If hair_count < min_hair_contours_to_process, it returns "No significant hairs found..."
            # and the file IS NOT SAVED by the function itself. We need to copy it.
            
            if "No significant hairs found" in msg or "original image skipped" in msg:
                if not os.path.exists(target_path_in_hair_removed_dir):
                    shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                    # print(f"Hair removal criteria not met for {filename} ({msg}). Original copied to target dir.")
                copied_original_count += 1
            elif os.path.exists(target_path_in_hair_removed_dir): # Implies hair removal happened and saved the file
                # print(f"Hair removal successful for {filename}: {msg}")
                processed_hair_removal_count += 1
            else: # Fallback: if file not saved for some reason by remove_and_save_hairs
                # print(f"Warning: Hair removal for {filename} - unexpected state. Message: '{msg}'. File not found at {target_path_in_hair_removed_dir}. Copying original as fallback.")
                shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
        
        except FileNotFoundError: # This can happen if original_image_path is wrong (unlikely in this loop)
            print(f"Error during hair removal for {filename}: Original image not found at {original_image_path}. Skipping this file.")
            hair_removal_error_count += 1
        except Exception as e_hair:
            print(f"Error during hair removal for {filename}: {e_hair}. Attempting to copy original.")
            hair_removal_error_count += 1
            try:
                if not os.path.exists(target_path_in_hair_removed_dir):
                    shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
                # print(f"Copied original {filename} to target dir due to hair removal error.")
            except Exception as e_copy:
                print(f"Failed to copy original {filename} after hair removal error: {e_copy}")
    
    print(f"Hair removal stage summary: Processed/inpainted: {processed_hair_removal_count - copied_original_count}, Originals copied (no/few hairs or error fallback): {copied_original_count}, Errors: {hair_removal_error_count}")
    print(f"Total images in hair-removed directory for feature extraction: {len(os.listdir(hair_removed_img_dir_path))}")

    # All subsequent feature extractions will use 'hair_removed_img_dir_path'
    feature_processing_dir = hair_removed_img_dir_path

    # --- Extract Asymmetry Features (Feature A) ---
    print(f"\nExtracting Asymmetry features from: {feature_processing_dir}")
    try:
        # Assuming extract_asymmetry_features processes images from folder_path
        # and if it needs masks, it handles finding them based on filenames (e.g., from mask_img_dir)
        df_A = extract_asymmetry_features(folder_path=feature_processing_dir, output_csv=None, visualize=False)
        if df_A.empty:
            print("Warning: Asymmetry feature extraction (feature_A) returned an empty DataFrame.")
        else:
            print(f"Asymmetry features extracted: {df_A.shape[0]} images, {df_A.shape[1]-1} features.")
    except Exception as e:
        print(f"Error during Asymmetry feature extraction: {e}")
        df_A = pd.DataFrame(columns=['filename'])

    # --- Extract Border Features (Feature B) ---
    print(f"\nExtracting Border features from: {feature_processing_dir}")
    try:
        df_B_raw = extract_border_features_from_folder(folder_path=feature_processing_dir, output_csv=None, visualize=False)
        if df_B_raw.empty:
            print("Warning: Border feature extraction (feature_B raw) returned an empty DataFrame.")
            df_B = pd.DataFrame(columns=['filename'])
        else:
            print(f"Raw Border features extracted: {df_B_raw.shape[0]} images.")
            df_B = calculate_border_score(df_B_raw)
            print(f"Border scores calculated. Total border features df: {df_B.shape[0]} images.")
            cols_to_drop_from_B = ['sobel_mean_safe', 'avg_contour_perimeter_safe', 'laplacian_mean_safe', 'avg_contour_area_safe']
            df_B = df_B.drop(columns=[col for col in cols_to_drop_from_B if col in df_B.columns], errors='ignore')
    except Exception as e:
        print(f"Error during Border feature extraction: {e}")
        df_B = pd.DataFrame(columns=['filename'])

    # --- Extract Color Features (Feature C) ---
    print(f"\nExtracting Color features from: {feature_processing_dir}")
    try:
        df_C = extract_feature_C(folder_path=feature_processing_dir, output_csv=None, normalize_colors=True, visualize=False)
        if df_C.empty:
            print("Warning: Color feature extraction (feature_C) returned an empty DataFrame.")
        else:
            print(f"Color features extracted: {df_C.shape[0]} images.")
    except Exception as e:
        print(f"Error during Color feature extraction: {e}")
        df_C = pd.DataFrame(columns=['filename'])

    # --- Extract Haralick Features (Feature H) ---
    print(f"\nExtracting Haralick features from: {feature_processing_dir}")
    try:
        # Use extract_haralick_features as imported
        df_H = extract_haralick_features(folder_path=feature_processing_dir, output_csv=None, visualize=False)
        if df_H.empty:
            print("Warning: Haralick feature extraction (feature_H) returned an empty DataFrame.")
        else:
            print(f"Haralick features extracted: {df_H.shape[0]} images.")
    except Exception as e:
        print(f"Error during Haralick feature extraction: {e}")
        df_H = pd.DataFrame(columns=['filename'])
        
    # --- Extract Blue Veil Features (Feature BV) ---
    print(f"\nExtracting Blue Veil features from: {feature_processing_dir}")
    try:
        df_BV = extract_feature_BV(folder_path=feature_processing_dir, output_csv=None, normalize_colors=True, visualize=False)
        if df_BV.empty:
            print("Warning: Blue Veil feature extraction (feature_BV) returned an empty DataFrame.")
        else:
            print(f"Blue Veil features extracted: {df_BV.shape[0]} images.")
    except Exception as e:
        print(f"Error during Blue Veil feature extraction: {e}")
        df_BV = pd.DataFrame(columns=['filename'])

    # --- Load Labels / Metadata ---
    # (This part remains largely the same, ensuring 'filename' matches processed images)
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
                metadata_df = None 
            else:
                if label_column_name != 'real_label':
                    raw_metadata_df.rename(columns={label_column_name: 'real_label'}, inplace=True)
                cancer_diagnoses = ["BCC", "SCC", "MEL"]
                raw_metadata_df['binary_target'] = raw_metadata_df['real_label'].apply(lambda x: 1 if x in cancer_diagnoses else 0)
                cols_to_keep_from_metadata = ['filename', 'real_label', 'binary_target']
                metadata_df = raw_metadata_df[cols_to_keep_from_metadata].copy()
                print(f"Metadata (filename, real_label, binary_target) selected. Shape: {metadata_df.shape[0]} entries.")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            metadata_df = None 
    else:
        print("\nNo metadata file provided or found. Proceeding without metadata.")

    # --- Merge DataFrames ---
    print("\nMerging feature DataFrames...")
    dataframes_to_merge = []
    if metadata_df is not None and not metadata_df.empty:
        dataframes_to_merge.append(metadata_df)
        print(f"metadata_df added. Shape: {metadata_df.shape}")
    
    # Add all feature dataframes that are not empty and have 'filename'
    feature_df_map = {"A": df_A, "B": df_B, "C": df_C, "H": df_H, "BV": df_BV}
    for name, df in feature_df_map.items():
        if not df.empty and 'filename' in df.columns:
            dataframes_to_merge.append(df)
            print(f"df_{name} added. Shape: {df.shape}")
        elif not df.empty:
             print(f"Skipping df_{name} in merge due to missing 'filename' column.")
        else:
             print(f"df_{name} is empty, not added.")

    if not dataframes_to_merge:
        print("No DataFrames with 'filename' column to merge. Exiting feature creation.")
        return pd.DataFrame()
    
    final_df = dataframes_to_merge[0]
    for i, df_to_merge in enumerate(dataframes_to_merge[1:]):
        # Ensure filename consistency (e.g., strip extensions if necessary, though ideally they match)
        # For example, if filenames in metadata have extensions and features don't, or vice-versa.
        # Assuming filenames are consistent as processed by feature extractors.
        final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner')
        if final_df.empty:
            print(f"CRITICAL WARNING: DataFrame became empty after merging with DataFrame {i+2} (likely df_{list(feature_df_map.keys())[i+1] if metadata_df is not None else list(feature_df_map.keys())[i]}). Check 'filename' consistency and content.")
            break 
            
    if final_df.empty:
        print("Resulting merged DataFrame is empty.")
    else:
        print(f"Merged DataFrame final shape: {final_df.shape}. Columns: {final_df.columns.tolist()}")

    # Ensure output directory exists from output_csv_path
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")
    
    return final_df

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- FEATURE DATASET CREATION (EXTENDED FEATURES) ---\n")
    
    if not original_img_dir or not output_csv_path:
        raise ValueError("original_img_dir and output_csv_path must be provided.")

    data_df = None
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, 
                                         labels_csv=labels_csv_path, recreate_features=recreate_features)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        try:
            data_df = pd.read_csv(output_csv_path)
        except Exception as e:
            print(f"Error loading existing dataset: {e}. Will attempt to recreate.")
            data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, 
                                             labels_csv=labels_csv_path, recreate_features=True) # Force recreate on load error

    if data_df is None or data_df.empty:
        print("Failed to create or load the feature dataset. Exiting.")
        return

    # ... (The rest of your main function for model training and evaluation remains the same) ...
    # Ensure this part correctly identifies feature columns, handles NaNs,
    # one-hot encodes if needed (e.g., 'c_dominant_channel'), and splits data.
    
    print("\n--- Merged Dataset Information ---")
    data_df.info(verbose=True, show_counts=True) 
    print("\nFirst 5 rows of the merged dataset:")
    print(data_df.head())
    
    if 'filename' not in data_df.columns:
        print("CRITICAL ERROR: 'filename' column is missing in the final DataFrame!")
        return
    if data_df['filename'].isnull().any():
        print("Warning: Some 'filename' entries are NaN. This usually indicates an issue in merging.")
        print(f"Number of NaN filenames: {data_df['filename'].isnull().sum()}")
    if data_df['filename'].duplicated().any():
        print(f"Warning: {data_df['filename'].duplicated().sum()} duplicate filenames found. Consolidating by keeping the first occurrence.")
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)


    print("\n--- MODEL TRAINING AND EVALUATION ---")
    
    if 'binary_target' not in data_df.columns:
        source_label_col = None
        if 'real_label' in data_df.columns: source_label_col = 'real_label'
        elif 'diagnostic' in data_df.columns: source_label_col = 'diagnostic'
        
        if source_label_col:
            print(f"Creating 'binary_target' from '{source_label_col}'.")
            cancer_diagnoses_map = ["BCC", "SCC", "MEL"]
            data_df['binary_target'] = data_df[source_label_col].apply(lambda x: 1 if x in cancer_diagnoses_map else 0)
            if source_label_col == 'diagnostic' and 'real_label' not in data_df.columns:
                data_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
        else:
            print("CRITICAL ERROR: Target label column ('binary_target', 'real_label', or 'diagnostic') not found.")
            return
    
    data_df.dropna(subset=['binary_target'], inplace=True) 
    if data_df.empty:
        print("CRITICAL ERROR: Dataset empty after dropping rows with missing 'binary_target'.")
        return

    data_df['label'] = data_df['binary_target'].astype(int)
    class_names_for_report = ['non-cancer', 'cancer'] 
    print(f"\nLabel distribution:\n{data_df['label'].value_counts(normalize=True)}")

    # One-hot encode 'c_dominant_channel' if it exists (from feature_C)
    if 'c_dominant_channel' in data_df.columns:
        print("\nOne-hot encoding 'c_dominant_channel'...")
        try:
            data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False)
            print("'c_dominant_channel' one-hot encoded.")
        except Exception as e_ohe:
            print(f"Error during one-hot encoding 'c_dominant_channel': {e_ohe}")
    
    potential_non_feature_cols = ['filename', 'real_label', 'binary_target', 'label', 'diagnostic', 
                                  'patient_id', 'lesion_id', 'smoke', 'drink', 
                                  'background_father', 'background_mother', 'age', 'pesticide', 
                                  'gender', 'skin_cancer_history', 'cancer_history', 'has_piped_water', 
                                  'has_sewage_system', 'fitspatrick', 'region', 'diameter_1', 
                                  'diameter_2', 'itch', 'grew', 'hurt', 'changed', 'bleed', 
                                  'elevation', 'biopsed'] 
    
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]
    
    if not feature_columns:
        print("CRITICAL ERROR: No feature columns identified. Cannot train model.")
        return
    print(f"\nUsing {len(feature_columns)} feature columns for training: {feature_columns[:15]}...") # Print first 15

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy() 
    current_filenames = data_df['filename'].copy()

    print("\nConverting features to numeric and handling NaNs/Infs...")
    for feat in feature_columns:
        x_all[feat] = pd.to_numeric(x_all[feat], errors='coerce')

    all_nan_cols = x_all.columns[x_all.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Columns became all NaN after numeric conversion and will be dropped: {all_nan_cols}")
        x_all = x_all.drop(columns=all_nan_cols)
        feature_columns = [col for col in feature_columns if col not in all_nan_cols]
        if not feature_columns:
            print("CRITICAL ERROR: All feature columns were dropped. Cannot train model.")
            return
    
    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean') # Or 'median' or 'most_frequent'
    
    if x_all.empty:
        print("CRITICAL ERROR: x_all DataFrame is empty before imputation. Cannot train model.")
        return

    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index)

    if len(x_all) == 0 or y_all.nunique() < 2:
        print(f"Skipping model training: Insufficient samples or classes. Samples: {len(x_all)}, Unique Labels: {y_all.nunique()}")
        return

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
    except ValueError as e_split:
        print(f"Error during data splitting: {e_split}. Check class distribution: \n{y_all.value_counts()}")
        return

    print(f"Training set: {len(x_train)}, Validation set: {len(x_val)}, Test set: {len(x_test)}")

    try:
        if x_train.empty or x_val.empty or y_train.nunique() < 2:
            print("Training/validation set empty or insufficient classes in training target. Skipping model training.")
            return

        best_model, best_model_name, best_val_acc = train_and_select_model(x_train, y_train, x_val, y_val)
        if best_model is None:
            print("No model was selected. Exiting.")
            return

        print("\n--- TEST PHASE ---")
        y_test_pred = best_model.predict(x_test)
        y_test_pred_proba = best_model.predict_proba(x_test) if hasattr(best_model, "predict_proba") else None

        test_acc = accuracy_score(y_test, y_test_pred)
        cm_labels_binary = [0, 1] 
        cm_display = confusion_matrix(y_test, y_test_pred, labels=cm_labels_binary)
        cls_report_dict = classification_report(y_test, y_test_pred, labels=cm_labels_binary, target_names=class_names_for_report, output_dict=True, zero_division=0)
        cls_report_str = classification_report(y_test, y_test_pred, labels=cm_labels_binary, target_names=class_names_for_report, zero_division=0)
        
        print(f"\nBest Model on Test Set: {best_model_name}, Test Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{cm_display}\nClassification Report:\n{cls_report_str}")

        # Reporting
        test_results_df = pd.DataFrame({
            'filename': filenames_test.values,
            'true_label_encoded': y_test.values,
            'predicted_label_encoded': y_test_pred,
            'true_label_text': y_test.map({0: 'non-cancer', 1: 'cancer'}).values,
            'predicted_label_text': pd.Series(y_test_pred).map({0: 'non-cancer', 1: 'cancer'}).values
        })
        if y_test_pred_proba is not None and y_test_pred_proba.shape[0] == len(y_test_pred) and y_test_pred_proba.shape[1] >= 2:
            test_results_df[f'proba_{class_names_for_report[0]}'] = y_test_pred_proba[:, 0]
            test_results_df[f'proba_{class_names_for_report[1]}'] = y_test_pred_proba[:, 1]
        
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        test_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_predictions_details_extended.csv")
        test_results_df.to_csv(test_details_csv_path, index=False)
        print(f"Detailed test predictions saved to {test_details_csv_path}")

        summary_report_data = {
            'model_name': best_model_name, 'validation_accuracy': best_val_acc, 'test_accuracy': test_acc,
            'num_training_samples': len(x_train), 'num_validation_samples': len(x_val), 'num_test_samples': len(x_test),
            'num_features_used': len(feature_columns),
        }
        for class_label_report in class_names_for_report: 
            if class_label_report in cls_report_dict:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    summary_report_data[f'{class_label_report}_{metric}_test'] = cls_report_dict[class_label_report][metric]
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in cls_report_dict:
                 for metric in ['precision', 'recall', 'f1-score']:
                    summary_report_data[f'{avg_type.replace(" ", "_")}_{metric}_test'] = cls_report_dict[avg_type][metric]

        results_summary_df = pd.DataFrame([summary_report_data])
        extended_result_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_extended.csv")
        results_summary_df.to_csv(extended_result_path, index=False)
        print(f"Summary model evaluation results for extended features saved to {extended_result_path}")

    except Exception as e_model:
        print(f"Error during model training/evaluation: {e_model}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks" 
    labels_csv_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"
    
    output_feature_csv_dir = "./result" 
    os.makedirs(output_feature_csv_dir, exist_ok=True)
    
    # New filename for the dataset with ABC + Haralick + Blue Veil features
    merged_csv_filename = "dataset_extended_ABC_H_BV_features.csv" 
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)
    
    # Model result filename can also be distinguished if needed
    model_result_filename = "model_evaluation_extended_features_summary.csv" 
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)
    
    try:
        # Set recreate_features=True to regenerate hair-removed images and all features
        # Set to False to load existing CSV if available
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True) 
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)