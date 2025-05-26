# In main_baseline.py
import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
            print(f"Raw metadata loaded. Columns: {raw_metadata_df.columns.tolist()}")
            # print(f"First 5 rows of raw metadata:\n{raw_metadata_df.head()}") # DEBUG

            if 'img_id' in raw_metadata_df.columns:
                print("Renaming 'img_id' to 'filename' in metadata.")
                raw_metadata_df = raw_metadata_df.rename(columns={'img_id': 'filename'})
            
            # Use 'diagnostic' as the source for 'real_label'
            label_column_name = 'diagnostic' # THIS IS THE KEY CHANGE FOR INPUT CSV

            if 'filename' not in raw_metadata_df.columns:
                print(f"ERROR: Metadata CSV must contain 'filename' (or 'img_id') column. Cannot proceed with metadata.")
                metadata_df = None 
            elif label_column_name not in raw_metadata_df.columns:
                print(f"ERROR: Metadata CSV must contain '{label_column_name}' column for model training. Cannot proceed with metadata.")
                metadata_df = None 
            else:
                print(f"'filename' and '{label_column_name}' columns found in metadata.")
                
                # Rename the source label column to 'real_label' for internal consistency
                if label_column_name != 'real_label':
                    raw_metadata_df.rename(columns={label_column_name: 'real_label'}, inplace=True)
                    print(f"Renamed '{label_column_name}' column to 'real_label' for internal consistency.")

                cancer_diagnoses = ["BCC", "SCC", "MEL"]
                raw_metadata_df['binary_target'] = raw_metadata_df['real_label'].apply(lambda x: 1 if x in cancer_diagnoses else 0)
                print("Binary target (0=non-cancer, 1=cancer) created from 'real_label' (originally '{label_column_name}').")
                print(f"Value counts for 'binary_target':\n{raw_metadata_df['binary_target'].value_counts(dropna=False)}")
                
                cols_to_keep_from_metadata = ['filename', 'real_label', 'binary_target']
                
                # Add other metadata columns that you want to use as features directly
                # Example: if 'age' from metadata was a feature, add 'age' to cols_to_keep_from_metadata
                # For now, just keeping the essential ones for labeling.

                missing_cols = [col for col in cols_to_keep_from_metadata if col not in raw_metadata_df.columns]
                if missing_cols:
                    print(f"ERROR: The following essential columns are missing from raw_metadata_df after processing: {missing_cols}")
                    metadata_df = None
                else:
                    metadata_df = raw_metadata_df[cols_to_keep_from_metadata].copy() 
                    print(f"Metadata (filename, real_label, binary_target) selected. Shape: {metadata_df.shape[0]} entries. Columns: {metadata_df.columns.tolist()}")
                    if metadata_df.empty:
                        print("Warning: metadata_df became empty after selecting columns. Check metadata CSV content and 'filename' consistency.")
                    # print(f"First 5 rows of selected metadata_df:\n{metadata_df.head()}") # DEBUG

        except Exception as e:
            print(f"Error loading metadata or creating binary_target: {e}")
            metadata_df = None 
    else:
        print("\nNo metadata file provided or file doesn't exist. Proceeding without metadata.")
        metadata_df = None

    # --- Merge DataFrames ---
    print("\nMerging feature DataFrames...")
    
    dataframes_to_merge = []
    if metadata_df is not None and not metadata_df.empty and 'filename' in metadata_df.columns:
        print(f"Attempting to add metadata_df to merge list. Shape: {metadata_df.shape}, Columns: {metadata_df.columns.tolist()}")
        dataframes_to_merge.append(metadata_df)
        print(f"metadata_df added to merge list. Current dataframes_to_merge count: {len(dataframes_to_merge)}")
    else:
        print("metadata_df was NOT added to merge list. This is a likely cause of missing label columns if labels_csv was provided.")
        if metadata_df is None:
            print("Reason: metadata_df is None (either not loaded, file not found, or error during processing).")
        elif metadata_df is not None and metadata_df.empty: 
            print("Reason: metadata_df is empty.")
        elif metadata_df is not None and 'filename' not in metadata_df.columns:
            print("Reason: metadata_df is missing 'filename' column.")
    
    feature_dfs_added = 0
    if not df_A.empty and 'filename' in df_A.columns:
        dataframes_to_merge.append(df_A)
        feature_dfs_added += 1
        print(f"df_A added. Shape: {df_A.shape}. Current dataframes_to_merge count: {len(dataframes_to_merge)}")
    elif not df_A.empty: print("Skipping df_A in merge due to missing 'filename' column or being empty.")
    else: print("df_A is empty, not added.")
        
    if not df_B.empty and 'filename' in df_B.columns:
        dataframes_to_merge.append(df_B)
        feature_dfs_added += 1
        print(f"df_B added. Shape: {df_B.shape}. Current dataframes_to_merge count: {len(dataframes_to_merge)}")
    elif not df_B.empty: print("Skipping df_B in merge due to missing 'filename' column or being empty.")
    else: print("df_B is empty, not added.")

    if not df_C.empty and 'filename' in df_C.columns:
        dataframes_to_merge.append(df_C)
        feature_dfs_added += 1
        print(f"df_C added. Shape: {df_C.shape}. Current dataframes_to_merge count: {len(dataframes_to_merge)}")
    elif not df_C.empty: print("Skipping df_C in merge due to missing 'filename' column or being empty.")
    else: print("df_C is empty, not added.")


    if not dataframes_to_merge:
        print("No DataFrames with 'filename' column to merge. Exiting feature creation.")
        return pd.DataFrame()
    
    print(f"Total DataFrames to merge: {len(dataframes_to_merge)}. Number of feature_dfs added: {feature_dfs_added}")
    if dataframes_to_merge:
        print(f"First DataFrame for merge has columns: {dataframes_to_merge[0].columns.tolist()} and shape {dataframes_to_merge[0].shape}")
        
    if len(dataframes_to_merge) == 1:
        print("Only one DataFrame available for merging. Using it as final_df.")
        final_df = dataframes_to_merge[0]
        if metadata_df is not None and final_df is metadata_df :
             print("This single DataFrame is metadata_df. No features were extracted or added.")
        elif feature_dfs_added == 1 and metadata_df is None: # Only one feature df, no metadata
             print("This single DataFrame is a feature DataFrame. Labels will be missing as metadata_df was not included or processed.")
        elif feature_dfs_added == 0 and metadata_df is not None: # Should have been caught by the first print.
            print("This single DataFrame is metadata_df, means no feature DataFrames were valid to add.")
        else: # Should not happen based on logic, but for safety
            print("Unclear state with a single DataFrame.")
             
    elif len(dataframes_to_merge) > 1:
        print(f"Proceeding with merge of {len(dataframes_to_merge)} DataFrames.")
        final_df = dataframes_to_merge[0]
        for i, df_to_merge in enumerate(dataframes_to_merge[1:]):
            
            common_filenames = pd.Series(list(set(final_df['filename']) & set(df_to_merge['filename'])))
            print(f"Merging with DataFrame {i+2} (shape {df_to_merge.shape}). Found {len(common_filenames)} common filenames.")
            if not common_filenames.empty:
                print(f"Sample common filenames: {common_filenames.head().tolist()}")
            else:
                print(f"WARNING: No common filenames for merge between current final_df and DataFrame {i+2}. This merge will result in an empty DataFrame if 'how=inner'.")

            final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner')
            print(f"Shape after merge {i+1}: {final_df.shape}. Columns: {final_df.columns.tolist()}")
            if final_df.empty:
                print(f"CRITICAL WARNING: DataFrame became empty after merging with DataFrame {i+2}. This is likely due to no common 'filename' values or inconsistent filename formats.")
                break
    else: 
        print("No DataFrames to merge (this means dataframes_to_merge list is empty). Returning empty DataFrame.")
        return pd.DataFrame()

    if final_df.empty:
        print("Resulting merged DataFrame is empty. This might be due to 'inner' merge and no common filenames or issues with feature extraction.")
    else:
        print(f"Merged DataFrame final shape: {final_df.shape}. Final columns: {final_df.columns.tolist()}")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")
    
    if not final_df.empty:
        expected_label_cols = ['real_label', 'binary_target']
        missing_label_cols_in_final = [col for col in expected_label_cols if col not in final_df.columns]
        if missing_label_cols_in_final:
            print(f"WARNING: The final merged DataFrame is MISSING these label columns: {missing_label_cols_in_final}")
        else:
            print(f"SUCCESS: 'real_label' and 'binary_target' columns are present in the final DataFrame.")
        
        # Exclude filename and all label-related columns for feature count
        non_feature_for_count = ['filename', 'real_label', 'binary_target', 'label'] # 'label' is created in main()
        feature_cols = [col for col in final_df.columns if col not in non_feature_for_count]
        print(f"Dataset contains {len(feature_cols)} potential feature columns.")
    
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
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True) # Add reset_index


    print("\n--- MODEL TRAINING AND EVALUATION ---")
    
    if 'binary_target' not in data_df.columns:
        source_label_col = None
        if 'real_label' in data_df.columns:
            source_label_col = 'real_label'
        elif 'diagnostic' in data_df.columns: 
            source_label_col = 'diagnostic'
            print("Warning: 'binary_target' and 'real_label' not found. Using 'diagnostic' to create binary labels.")
        
        if source_label_col:
            print(f"Creating 'binary_target' from '{source_label_col}' column as it was missing.")
            cancer_diagnoses_map = ["BCC", "SCC", "MEL"]
            data_df['binary_target'] = data_df[source_label_col].apply(lambda x: 1 if x in cancer_diagnoses_map else 0)
            if source_label_col == 'diagnostic' and 'real_label' not in data_df.columns:
                data_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
        else:
            print("CRITICAL ERROR: Neither 'binary_target', 'real_label', nor 'diagnostic' column found in the dataset. Cannot proceed.")
            return
    
    data_df.dropna(subset=['binary_target'], inplace=True) 
    if data_df.empty:
        print("CRITICAL ERROR: Dataset became empty after dropping rows with missing 'binary_target' labels.")
        return

    data_df['label'] = data_df['binary_target'].astype(int)
    class_names_for_report = ['non-cancer', 'cancer'] 
    print(f"\nBinary classification: 0 -> {class_names_for_report[0]}, 1 -> {class_names_for_report[1]}")
    print(f"Label distribution:\n{data_df['label'].value_counts(normalize=True)}")

    if 'real_label' not in data_df.columns and 'diagnostic' in data_df.columns:
        print("Copying 'diagnostic' to 'real_label' for reporting purposes as 'real_label' was missing.")
        data_df['real_label'] = data_df['diagnostic']
    elif 'real_label' not in data_df.columns:
        print("Warning: 'real_label' (or 'diagnostic') not found for original diagnosis text in reporting.")
        # Create a placeholder 'real_label' if absolutely necessary for reporting structure,
        # though ideally, it should come from the original data.
        data_df['real_label'] = data_df['label'].map({0: 'derived_non-cancer', 1: 'derived_cancer'})


    if 'c_dominant_channel' in data_df.columns:
        print("\nOne-hot encoding 'c_dominant_channel'...")
        data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False)
    
    potential_non_feature_cols = ['filename', 'real_label', 'binary_target', 'label', 
                                  'diagnostic', 
                                  'patient_id', 'lesion_id', 'smoke', 'drink', 
                                  'background_father', 'background_mother', 'age', 'pesticide', 
                                  'gender', 'skin_cancer_history', 'cancer_history', 'has_piped_water', 
                                  'has_sewage_system', 'fitspatrick', 'region', 'diameter_1', 
                                  'diameter_2', 'itch', 'grew', 'hurt', 'changed', 'bleed', 
                                  'elevation', 'biopsed'] 
    
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols and col != 'label_text_binary']
    
    if not feature_columns:
        print("CRITICAL ERROR: No feature columns identified after exclusions. Cannot train model.")
        return
    print(f"\nUsing {len(feature_columns)} feature columns for training: {feature_columns}")

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy() 
    current_filenames = data_df['filename'].copy()

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
    
    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index)

    if len(x_all) == 0:
        print("Skipping model training: No samples remaining after data cleaning.")
        return
    if y_all.nunique() < 2:
        print(f"Skipping model training: Not enough unique classes in labels for stratified split or training. Unique labels: {y_all.unique()}")
        return

    print(f"\nSplitting data into train, validation, and test sets (Total samples: {len(x_all)})...")
    try:
        # Important: Ensure the arrays passed to train_test_split have consistent indexing if they are pandas objects
        # If x_all, y_all, current_filenames are derived from data_df that had its index reset after drop_duplicates, this should be fine.
        x_train, x_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
            x_all.reset_index(drop=True), 
            y_all.reset_index(drop=True), 
            current_filenames.reset_index(drop=True), 
            test_size=0.4, random_state=42, stratify=y_all.reset_index(drop=True) # Stratify needs array-like without mixed indices
        )
        x_val, x_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
            x_temp.reset_index(drop=True), 
            y_temp.reset_index(drop=True), 
            filenames_temp.reset_index(drop=True), 
            test_size=0.5, random_state=42, stratify=y_temp.reset_index(drop=True)
        )
    except ValueError as e_split:
        print(f"Error during data splitting: {e_split}. This might be due to too few samples in some classes.")
        print(f"Class distribution in y_all: \n{y_all.value_counts()}")
        return

    print(f"Training set size: {len(x_train)}")
    print(f"Validation set size: {len(x_val)}")
    print(f"Test set size: {len(x_test)}")

    try:
        if x_train.empty or x_val.empty:
            print("Training or validation set is empty before calling train_and_select_model. Skipping.")
            return
        if y_train.nunique() < 2 : # Allow y_val to have 1 class for now, models_evaluation handles it.
            print(f"Training target has less than 2 unique classes. y_train unique: {y_train.nunique()}. Skipping model training.")
            return

        best_model, best_model_name, best_val_acc = train_and_select_model(x_train, y_train, x_val, y_val)

        if best_model is None:
            print("No model was selected. Exiting.")
            return

        print("\n--- TEST PHASE ---")
        y_test_pred = best_model.predict(x_test)
        
        if hasattr(best_model, "predict_proba"):
            y_test_pred_proba = best_model.predict_proba(x_test)
        else:
            print(f"Warning: Model {best_model_name} does not have predict_proba. Probabilities will be estimated.")
            num_classes_binary = len(class_names_for_report) 
            y_test_pred_proba = np.zeros((len(y_test_pred), num_classes_binary))
            for i, pred_label in enumerate(y_test_pred):
                y_test_pred_proba[i, pred_label] = 1.0 

        test_acc = accuracy_score(y_test, y_test_pred)
        cm_labels_binary = [0, 1] 
        cm_display = confusion_matrix(y_test, y_test_pred, labels=cm_labels_binary)
        
        cls_report_dict = classification_report(y_test, y_test_pred, labels=cm_labels_binary, target_names=class_names_for_report, output_dict=True, zero_division=0)
        cls_report_str = classification_report(y_test, y_test_pred, labels=cm_labels_binary, target_names=class_names_for_report, zero_division=0)
        
        print(f"\nBest Model on Test Set: {best_model_name}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix (Test Set) - Labels {class_names_for_report}:\n{cm_display}")
        print(f"Classification Report (Test Set):\n{cls_report_str}")

        # 8. Reporting - THIS IS THE CORRECTED SECTION
        filenames_array = filenames_test.reset_index(drop=True).values # Use .values after reset_index
        true_labels_encoded_array = y_test.reset_index(drop=True).values # Use .values after reset_index
        # y_test_pred is already a numpy array
        
        true_labels_text_array = y_test.reset_index(drop=True).map({0: 'non-cancer', 1: 'cancer'}).values
        predicted_labels_text_array = pd.Series(y_test_pred).map({0: 'non-cancer', 1: 'cancer'}).values # y_test_pred is numpy array, so Series has fresh index

        test_results_df = pd.DataFrame({
            'filename': filenames_array,
            'true_label_encoded': true_labels_encoded_array,
            'predicted_label_encoded': y_test_pred,
            'true_label_text': true_labels_text_array,
            'predicted_label_text': predicted_labels_text_array
        })
        
        # Add probabilities
        # Ensure y_test_pred_proba rows match the length of other arrays
        if y_test_pred_proba.shape[0] == len(filenames_array):
            if y_test_pred_proba.shape[1] == len(class_names_for_report): 
                test_results_df[f'proba_{class_names_for_report[0]}'] = y_test_pred_proba[:, 0]
                test_results_df[f'proba_{class_names_for_report[1]}'] = y_test_pred_proba[:, 1]
            else:
                print(f"Warning: Mismatch in probability array columns ({y_test_pred_proba.shape[1]}) and class_names_for_report length ({len(class_names_for_report)})")
                test_results_df[f'proba_{class_names_for_report[0]}'] = 0.0 
                test_results_df[f'proba_{class_names_for_report[1]}'] = 0.0
        else:
            print(f"Warning: Mismatch in y_test_pred_proba rows ({y_test_pred_proba.shape[0]}) and expected test set size ({len(filenames_array)})")
            # Pad or truncate y_test_pred_proba if necessary, or assign NaNs/zeros
            # For simplicity, assigning NaNs if there's a length mismatch for probabilities
            for cn in class_names_for_report:
                test_results_df[f'proba_{cn}'] = np.nan


        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        test_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_predictions_details.csv")
        test_results_df.to_csv(test_details_csv_path, index=False)
        print(f"Detailed test predictions saved to {test_details_csv_path}")

        summary_report_data = {
            'model_name': best_model_name,
            'validation_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'num_training_samples': len(x_train),
            'num_validation_samples': len(x_val),
            'num_test_samples': len(x_test),
            'num_features_used': len(feature_columns),
        }
        for class_label_report in class_names_for_report: 
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

# THE if __name__ == "__main__": BLOCK REMAINS THE SAME
if __name__ == "__main__":
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks" 
    labels_csv_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"
    
    output_feature_csv_dir = "./result" 
    os.makedirs(output_feature_csv_dir, exist_ok=True)
    
    merged_csv_filename = "dataset_baseline_features_binary_target.csv" 
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)
    
    model_result_filename = "model_evaluation_binary_summary.csv" 
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)
    
    try:
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)