import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Load the dataset
DATA_PATH = Path("outputs/features_dataset.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError("Run train_pipeline.py first to generate features_dataset.csv")

df_feat = pd.read_csv(DATA_PATH)

X = df_feat.drop(columns=["label", "group"])
y = df_feat["label"]
groups = df_feat["group"]

# 2. Session-based split to prevent data leakage
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 3. Define the Models (Notice StandardScaler for SVM & KNN!)
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, 
        random_state=42, 
        class_weight="balanced_subsample", 
        n_jobs=-1
    ),
    "SVM (RBF Kernel)": make_pipeline(
        StandardScaler(), 
        SVC(kernel='rbf', class_weight='balanced', random_state=42)
    ),
    "KNN (K=5)": make_pipeline(
        StandardScaler(), 
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    )
}

# 4. Train and Evaluate
results = []

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Store for summary
    results.append({"Model": name, "Accuracy": round(acc, 4)})

print(f"\n{'='*40}")
print("SUMMARY TABLE")
print(pd.DataFrame(results).to_markdown(index=False))
