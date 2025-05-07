from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_select_model(x_train, y_train, x_val, y_val):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    best_model = None
    best_model_name = None
    best_val_acc = 0

    print("\n--- VALIDATION PHASE ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        acc = accuracy_score(y_val, val_pred)
        print(f"Validation Accuracy for {name}: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            best_model = model
            best_model_name = name

    print(f"\n✅ Best model: {best_model_name} (Val Accuracy: {best_val_acc:.4f})")
    return best_model, best_model_name
