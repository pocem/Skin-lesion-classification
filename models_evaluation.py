
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix # confusion_matrix is not used here but good to keep if you expand

def train_and_select_model(x_train, y_train, x_val, y_val):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'), # Added solver to help convergence for smaller datasets
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    best_model = None
    best_model_name = None
    best_val_acc = 0.0 # Initialize as float

    print("\n--- VALIDATION PHASE ---")
    if x_train.empty or x_val.empty:
        print("Warning: Training or validation data is empty. Skipping model training.")
        return None, "No Model", 0.0


    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
            acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy for {name}: {acc:.4f}")

            if acc > best_val_acc:
                best_val_acc = acc
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"Error training or validating {name}: {e}")
            # Optionally, continue to the next model or handle more gracefully

    if best_model_name:
        print(f"\n✅ Best model from validation: {best_model_name} (Val Accuracy: {best_val_acc:.4f})")
    else:
        print("\nNo model was successfully trained or selected during validation.")
        # Return dummy values to prevent downstream errors if main_baseline expects them
        # Or raise an error if no model can be selected
        return None, "No Model Selected", 0.0


    return best_model, best_model_name, best_val_acc