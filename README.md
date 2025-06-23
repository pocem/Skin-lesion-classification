# Skin Lesion Classification with Handcrafted Features

This project implements a complete machine learning pipeline to classify skin lesions as either **Malignant** or **Benign**. It uses a Random Forest classifier trained on a set of handcrafted features inspired by the dermatological ABCDE rules (Asymmetry, Border, Color).

This work is a continuation and refinement of a university group project, focusing on creating a robust, reproducible, and optimized classification model.

## Project Structure

- **`/data/`**: Should contain the `metadata_matched.csv` file with lesion diagnostics.
- **`/util/`**: Contains the Python scripts for feature extraction (`feature_A.py`, `feature_B.py`, etc.).
- **`build_dataset.py`**: The first script to run. It extracts all features and merges them with metadata to create the final, clean dataset.
- **`training.py`**: The second script to run. It loads the clean dataset and handles all model training, tuning, and evaluation.
- **`/model_results/`**: The output directory where evaluation reports and prediction files are saved.

---

## Data

The model was developed using the **PAD-UFES-20 dataset**. This dataset provides:
1.  Original clinical images of skin lesions.
2.  Binary segmentation masks that isolate the lesion area.
3.  A metadata file containing the ground truth `diagnostic` for each lesion.

Due to size and privacy considerations, the image data is not included in this repository.

---

## Methodology

The project follows a two-stage classical machine learning pipeline:

### 1. Feature Engineering & Dataset Creation

The `build_binary_dataset.py` script performs an automated ETL (Extract, Transform, Load) process:

-   **Asymmetry (A):** Calculates geometric and PCA-based asymmetry from the lesion masks.
-   **Border (B):** Extracts features related to the irregularity and texture of the lesion's border.
-   **Color (C):** Analyzes color variance, hue, saturation, and asymmetry from the original images.
-   **Blue-Veil (BV):** A specialized feature to detect the presence and area of a blue-whitish veil, a key clinical indicator.

These features are then merged with the ground truth labels from the metadata. The six original lesion types are grouped into a single binary target:
-   **Malignant (1):** BCC, SCC, MEL, ACK
-   **Benign (0):** NEV, SEK

### 2. Model Training and Evaluation

The `train_binary_model.py` script handles the machine learning workflow:

-   **Classifier:** A `RandomForestClassifier` is used for its robustness and performance.
-   **Class Imbalance:** **SMOTE** (Synthetic Minority Over-sampling TEchnique) is applied to the training set to correct the imbalance between the benign and malignant classes, ensuring the model does not ignore the minority class.
-   **Hyperparameter Tuning:** `GridSearchCV` automatically searches for the optimal model parameters (`n_estimators`, `max_depth`, etc.) to maximize performance.
-   **Evaluation:** The model is trained and validated on a 70/15 split of the data, with the final performance reported on a completely held-out 15% **test set**.

---

## How to Use This Project

To reproduce the results, follow these steps:

### 1. Prerequisites

-   Python 3.8+
-   A prepared dataset with the following structure:
    -   `matched_data/images/` (containing original lesion images)
    -   `matched_data/masks/` (containing corresponding binary masks)
-   Create a `data` folder in the project root and place your metadata CSV inside it, named `metadata_matched.csv`.

### 2. Installation

Clone the repository and install the required Python libraries:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt