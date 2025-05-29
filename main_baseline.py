import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier # Added RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
    # from models_evaluation import train_and_select_model # Commented out as we'll use RF directly
except ImportError as e:
    print(f"Error: Could not import custom feature modules: {e}")
    print("Please ensure feature_A.py, feature_B.py, feature_C.py are in the 'util' directory (or adjust import paths).")
    # print("Ensure models_evaluation.py is in the same directory or Python path if using train_and_select_model.")
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


    metadata_df = None
    if labels_csv and exists(labels_csv):
        print(f"\nLoading metadata from {labels_csv}")
        try:
            raw_metadata_df = pd.read_csv(labels_csv)
            print(f"Raw metadata loaded. Columns: {raw_metadata_df.columns.tolist()}")


            if 'img_id' in raw_metadata_df.columns:
                print("Renaming 'img_id' to 'filename' in metadata.")
                raw_metadata_df = raw_metadata_df.rename(columns={'img_id': 'filename'})

            label_column_name = 'diagnostic'

            if 'filename' not in raw_metadata_df.columns:
                print(f"ERROR: Metadata CSV must contain 'filename' (or 'img_id') column. Cannot proceed with metadata.")
                metadata_df = None
            elif label_column_name not in raw_metadata_df.columns:
                print(f"ERROR: Metadata CSV must contain '{label_column_name}' column for model training. Cannot proceed with metadata.")
                metadata_df = None
            else:
                print(f"'filename' and '{label_column_name}' columns found in metadata.")


                if label_column_name != 'real_label':
                    raw_metadata_df.rename(columns={label_column_name: 'real_label'}, inplace=True)
                    print(f"Renamed '{label_column_name}' column to 'real_label' for internal consistency.")

                cancer_diagnoses = ["BCC", "SCC", "MEL"]
                raw_metadata_df['binary_target'] = raw_metadata_df['real_label'].apply(lambda x: 1 if x in cancer_diagnoses else 0)
                print("Binary target (0=non-cancer, 1=cancer) created from 'real_label' (originally '{label_column_name}').")
                print(f"Value counts for 'binary_target':\n{raw_metadata_df['binary_target'].value_counts(dropna=False)}")

                cols_to_keep_from_metadata = ['filename', 'real_label', 'binary_target']

                missing_cols = [col for col in cols_to_keep_from_metadata if col not in raw_metadata_df.columns]
                if missing_cols:
                    print(f"ERROR: The following essential columns are missing from raw_metadata_df after processing: {missing_cols}")
                    metadata_df = None
                else:
                    metadata_df = raw_metadata_df[cols_to_keep_from_metadata].copy()
                    print(f"Metadata (filename, real_label, binary_target) selected. Shape: {metadata_df.shape[0]} entries. Columns: {metadata_df.columns.tolist()}")
                    if metadata_df.empty:
                        print("Warning: metadata_df became empty after selecting columns. Check metadata CSV content and 'filename' consistency.")

        except Exception as e:
            print(f"Error loading metadata or creating binary_target: {e}")
            metadata_df = None
    else:
        print("\nNo metadata file provided or file doesn't exist. Proceeding without metadata.")
        metadata_df = None


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
        elif feature_dfs_added == 1 and metadata_df is None:
             print("This single DataFrame is a feature DataFrame. Labels will be missing as metadata_df was not included or processed.")
        elif feature_dfs_added == 0 and metadata_df is not None:
            print("This single DataFrame is metadata_df, means no feature DataFrames were valid to add.")
        else:
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


        non_feature_for_count = ['filename', 'real_label', 'binary_target', 'label']
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
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)


    # --- DATA PREPARATION FOR MODELING ---
    print("\n--- DATA PREPARATION FOR MODELING ---")

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
    current_filenames = data_df['filename'].copy() # Keep track of filenames corresponding to x_all

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
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index) # Keep original index for iloc

    if len(x_all) == 0:
        print("Skipping model training: No samples remaining after data cleaning.")
        return
    if y_all.nunique() < 2:
        print(f"Skipping model training: Not enough unique classes in labels for stratified split or training. Unique labels: {y_all.unique()}")
        return

    # --- K-FOLD CROSS-VALIDATION with RANDOM FOREST ONLY ---
    N_SPLITS = 5
    min_class_count = y_all.value_counts().min()
    if N_SPLITS > min_class_count:
        print(f"Warning: N_SPLITS ({N_SPLITS}) is greater than the number of samples in the smallest class ({min_class_count}).")
        print(f"Reducing N_SPLITS to {min_class_count} to allow stratified splitting.")
        N_SPLITS = min_class_count
        if N_SPLITS < 2:
             print("Smallest class has less than 2 samples. Cannot perform K-Fold CV. Exiting model training.")
             return

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results_list = []
    all_test_predictions_df = pd.DataFrame()

    print(f"\n--- {N_SPLITS}-FOLD CROSS-VALIDATION (Random Forest Only) ---")

    for fold_num, (dev_indices, test_indices) in enumerate(skf.split(x_all, y_all)):
        print(f"\n--- FOLD {fold_num + 1}/{N_SPLITS} ---")

        x_dev_fold = x_all.iloc[dev_indices]
        y_dev_fold = y_all.iloc[dev_indices]
        x_test_fold = x_all.iloc[test_indices]
        y_test_fold = y_all.iloc[test_indices]
        filenames_test_fold = current_filenames.iloc[test_indices]

        try:
            x_train_inner, x_val_inner, y_train_inner, y_val_inner = train_test_split(
                x_dev_fold, y_dev_fold, test_size=0.25, random_state=42, stratify=y_dev_fold
            )
        except ValueError as e_split_inner:
            print(f"Error during inner data splitting for fold {fold_num + 1}: {e_split_inner}.")
            print(f"Class distribution in y_dev_fold: \n{y_dev_fold.value_counts()}")
            continue

        print(f"Fold {fold_num + 1}: Train_inner size: {len(x_train_inner)}, Val_inner size: {len(x_val_inner)}, Test_fold size: {len(x_test_fold)}")

        if x_train_inner.empty or x_val_inner.empty or y_train_inner.nunique() < 2:
            print(f"Fold {fold_num + 1}: Training_inner or validation_inner set is empty or has insufficient classes. Skipping this fold.")
            continue

        try:
            # --- Using RandomForestClassifier directly ---
            model_fold = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model_fold.fit(x_train_inner, y_train_inner)
            model_name_fold = "RandomForestClassifier"

            # Evaluate on inner validation set
            y_val_inner_pred = model_fold.predict(x_val_inner)
            val_acc_inner_fold = accuracy_score(y_val_inner, y_val_inner_pred)
            print(f"Fold {fold_num + 1} - Inner Validation Accuracy (RF): {val_acc_inner_fold:.4f}")
            # --- End of RandomForestClassifier direct usage ---

            print(f"\nFold {fold_num + 1} - Test Phase on test_fold data...")
            y_test_pred_fold = model_fold.predict(x_test_fold)

            y_test_pred_proba_fold = None
            if hasattr(model_fold, "predict_proba"):
                y_test_pred_proba_fold = model_fold.predict_proba(x_test_fold)
            else: # Should not happen for RF, but good to keep
                print(f"Warning: Model {model_name_fold} in fold {fold_num + 1} does not have predict_proba.")
                num_classes_binary_fold = len(class_names_for_report)
                y_test_pred_proba_fold = np.zeros((len(y_test_pred_fold), num_classes_binary_fold))
                for i, pred_label in enumerate(y_test_pred_fold):
                    if 0 <= pred_label < num_classes_binary_fold:
                        y_test_pred_proba_fold[i, pred_label] = 1.0

            test_acc_fold = accuracy_score(y_test_fold, y_test_pred_fold)
            cm_labels_binary_fold = [0, 1]

            cls_report_dict_fold = classification_report(
                y_test_fold, y_test_pred_fold, labels=cm_labels_binary_fold,
                target_names=class_names_for_report, output_dict=True, zero_division=0
            )
            cls_report_str_fold = classification_report(
                y_test_fold, y_test_pred_fold, labels=cm_labels_binary_fold,
                target_names=class_names_for_report, zero_division=0
            )

            print(f"Fold {fold_num + 1} - Model: {model_name_fold}")
            print(f"Fold {fold_num + 1} - Test Accuracy on test_fold: {test_acc_fold:.4f}")
            print(f"Fold {fold_num + 1} - Confusion Matrix (test_fold):\n{confusion_matrix(y_test_fold, y_test_pred_fold, labels=cm_labels_binary_fold)}")
            print(f"Fold {fold_num + 1} - Classification Report (test_fold):\n{cls_report_str_fold}")

            fold_summary = {
                'fold': fold_num + 1,
                'model_name': model_name_fold,
                'validation_accuracy_inner': val_acc_inner_fold, # Using inner val acc
                'test_accuracy_fold': test_acc_fold,
                'num_training_samples_inner': len(x_train_inner),
                'num_validation_samples_inner': len(x_val_inner),
                'num_test_samples_fold': len(x_test_fold),
                'num_features_used': len(feature_columns)
            }
            for class_label_report_fold in class_names_for_report:
                if class_label_report_fold in cls_report_dict_fold:
                    for metric in ['precision', 'recall', 'f1-score', 'support']:
                        fold_summary[f'{class_label_report_fold}_{metric}_test_fold'] = cls_report_dict_fold[class_label_report_fold][metric]
            for avg_type_fold in ['macro avg', 'weighted avg']:
                if avg_type_fold in cls_report_dict_fold:
                    for metric in ['precision', 'recall', 'f1-score']:
                        fold_summary[f'{avg_type_fold.replace(" ", "_")}_{metric}_test_fold'] = cls_report_dict_fold[avg_type_fold][metric]
            fold_results_list.append(fold_summary)

            current_fold_predictions_df = pd.DataFrame({
                'fold': fold_num + 1,
                'filename': filenames_test_fold.reset_index(drop=True).values,
                'true_label_encoded': y_test_fold.reset_index(drop=True).values,
                'predicted_label_encoded': y_test_pred_fold,
                'true_label_text': y_test_fold.reset_index(drop=True).map({0: 'non-cancer', 1: 'cancer'}).values,
                'predicted_label_text': pd.Series(y_test_pred_fold).map({0: 'non-cancer', 1: 'cancer'}).values
            })
            if y_test_pred_proba_fold is not None and y_test_pred_proba_fold.shape[0] == len(filenames_test_fold):
                if y_test_pred_proba_fold.shape[1] >= len(class_names_for_report):
                    current_fold_predictions_df[f'proba_{class_names_for_report[0]}'] = y_test_pred_proba_fold[:, 0]
                    current_fold_predictions_df[f'proba_{class_names_for_report[1]}'] = y_test_pred_proba_fold[:, 1]
                else:
                    print(f"Warning: Fold {fold_num+1} Mismatch in probability array columns for detailed predictions.")
                    current_fold_predictions_df[f'proba_{class_names_for_report[0]}'] = np.nan
                    current_fold_predictions_df[f'proba_{class_names_for_report[1]}'] = np.nan
            else:
                print(f"Warning: Fold {fold_num+1} Mismatch/Missing probability array for detailed predictions.")
                current_fold_predictions_df[f'proba_{class_names_for_report[0]}'] = np.nan
                current_fold_predictions_df[f'proba_{class_names_for_report[1]}'] = np.nan

            all_test_predictions_df = pd.concat([all_test_predictions_df, current_fold_predictions_df], ignore_index=True)

        except Exception as e_model_fold:
            print(f"Error during model training/evaluation for fold {fold_num + 1}: {e_model_fold}")
            import traceback
            traceback.print_exc()

    # --- AGGREGATE RESULTS FROM K-FOLD CV ---
    if not fold_results_list:
        print("No folds were successfully processed. Cannot generate CV summary. Exiting.")
        return

    print("\n\n--- K-FOLD CROSS-VALIDATION SUMMARY (Random Forest Only) ---")
    cv_summary_df = pd.DataFrame(fold_results_list)

    avg_metrics_summary = {
        'model_type_fixed': "RandomForestClassifier", # Added to indicate model used
        'num_folds_processed': len(cv_summary_df),
        'num_features_used': cv_summary_df['num_features_used'].iloc[0] if not cv_summary_df.empty else 0
    }

    metrics_to_process = ['test_accuracy_fold', 'validation_accuracy_inner'] # Added inner validation acc
    for cl_label in class_names_for_report:
        for metric in ['precision', 'recall', 'f1-score', 'support']:
            metrics_to_process.append(f'{cl_label}_{metric}_test_fold')
    for avg_type in ['macro avg', 'weighted avg']:
        for metric in ['precision', 'recall', 'f1-score']:
            metrics_to_process.append(f'{avg_type.replace(" ", "_")}_{metric}_test_fold')

    for metric_col in metrics_to_process:
        if metric_col in cv_summary_df.columns:
            base_metric_name = metric_col.replace('_test_fold', '').replace('_inner', '_inner_val') # Clarify inner val
            if 'support' in metric_col:
                avg_metrics_summary[f'total_{base_metric_name}'] = cv_summary_df[metric_col].sum()
                avg_metrics_summary[f'mean_{base_metric_name}'] = cv_summary_df[metric_col].mean()
                avg_metrics_summary[f'std_{base_metric_name}'] = cv_summary_df[metric_col].std()
                print(f"Total {base_metric_name}: {avg_metrics_summary[f'total_{base_metric_name}']}")
                print(f"Mean {base_metric_name} per fold: {avg_metrics_summary[f'mean_{base_metric_name}']:.4f} +/- {avg_metrics_summary[f'std_{base_metric_name}']:.4f}")
            else:
                avg_metrics_summary[f'mean_{base_metric_name}'] = cv_summary_df[metric_col].mean()
                avg_metrics_summary[f'std_{base_metric_name}'] = cv_summary_df[metric_col].std()
                print(f"Mean {base_metric_name}: {avg_metrics_summary[f'mean_{base_metric_name}']:.4f} +/- {avg_metrics_summary[f'std_{base_metric_name}']:.4f}")

    print(f"\nOverall Average Inner Validation Accuracy across {N_SPLITS} folds: {avg_metrics_summary.get('mean_validation_accuracy_inner_val', np.nan):.4f} +/- {avg_metrics_summary.get('std_validation_accuracy_inner_val', np.nan):.4f}")
    print(f"Overall Average Test Accuracy across {N_SPLITS} folds: {avg_metrics_summary.get('mean_test_accuracy', np.nan):.4f} +/- {avg_metrics_summary.get('std_test_accuracy', np.nan):.4f}")


    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    fold_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_CV_fold_details.csv")
    cv_summary_df.to_csv(fold_details_csv_path, index=False)
    print(f"Detailed K-Fold CV results per fold saved to {fold_details_csv_path}")

    all_predictions_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_CV_all_predictions.csv")
    all_test_predictions_df.to_csv(all_predictions_csv_path, index=False)
    print(f"All test predictions from K-Fold CV saved to {all_predictions_csv_path}")

    aggregated_summary_df = pd.DataFrame([avg_metrics_summary])
    aggregated_summary_df.to_csv(result_path, index=False)
    print(f"Aggregated K-Fold CV summary report saved to {result_path}")


if __name__ == "__main__":
    original_img_dir = r"C:\Users\misog\SCHOOL\2nd semester\Projects in Data Science\matched_pairs\images"
    mask_img_dir = r"C:\Users\misog\SCHOOL\2nd semester\Projects in Data Science\masks"
    labels_csv_path = r"C:\Users\misog\SCHOOL\2nd semester\Projects in Data Science\final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"

    # Suggestion: Modify output directory/filenames to indicate Random Forest only
    output_feature_csv_dir = "./result_CV_baseline_RF_only"
    os.makedirs(output_feature_csv_dir, exist_ok=True)

    merged_csv_filename = "dataset_baseline_features_binary_target_CV_RF_only.csv"
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)

    model_result_filename = "model_evaluation_binary_CV_RF_only_summary.csv"
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)

    try:
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)