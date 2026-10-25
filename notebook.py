# # Predictive Maintenance: Random Forest vs SVM vs KNN vs Decision Tree
# Dataset assumed: AI4I 2020 Predictive Maintenance (ai4i2020.csv)
# Run cell by cell in Jupyter / VS Code / Colab, or as a normal script.

# %% 1. LIBRARIES
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, average_precision_score
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# %% 2. LOCATE THE CSV (Kaggle)
import os, glob, re

KAGGLE_PATH = "/kaggle/input/datasets/rajpranjal671431/predictive-maintenance-data"

if os.path.isfile(KAGGLE_PATH):
    DATA_PATH = KAGGLE_PATH
elif os.path.isdir(KAGGLE_PATH):
    csvs = sorted(glob.glob(os.path.join(KAGGLE_PATH, "**", "*.csv"), recursive=True))
    print("CSV files found:", csvs)
    DATA_PATH = csvs[0]          # change the index if the folder has several CSVs
else:
    print("Path not found. Files available under /kaggle/input:")
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            print(os.path.join(root, f))
    raise FileNotFoundError("Copy the correct CSV path from the list above into KAGGLE_PATH")

print("Using:", DATA_PATH)

# %% 3. LOAD DATA
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
df.head()

# %% 3B. CONFIGURATION (auto-detected; override manually if the printout looks wrong)
def clean_name(c):
    """Remove [units], trim, replace spaces so names are safe for all libraries."""
    return re.sub(r"\s+", "_", re.sub(r"\[.*?\]", "", str(c)).strip())

# --- Target column ---
TARGET = None   # <- set manually, e.g. TARGET = "Machine failure", if auto-detect fails
if TARGET is None:
    candidates = ["machine failure", "machine_failure", "failure", "target", "label",
                  "fail", "failure_flag", "breakdown", "faulty", "y"]
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            TARGET = lower_map[cand]
            break
if TARGET is None:
    raise ValueError(f"Could not auto-detect target. Set TARGET manually. Columns: {list(df.columns)}")

# --- Leakage columns: failure-cause / failure-type columns that encode the target ---
LEAK_NAMES = {"twf", "hdf", "pwf", "osf", "rnf", "failure type", "failure_type"}
LEAKAGE_COLS = [c for c in df.columns if c.lower().strip() in LEAK_NAMES and c != TARGET]

# --- ID columns: no predictive value ---
ID_NAMES = {"udi", "id", "uid", "product id", "product_id", "machine id", "machine_id", "index"}
ID_COLS = [c for c in df.columns if c != TARGET and (
    c.lower().strip() in ID_NAMES
    or (not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 0.5 * len(df)))]

# --- Categorical columns ---
CATEGORICAL = [c for c in df.select_dtypes(exclude=["number", "bool"]).columns
               if c not in ID_COLS + LEAKAGE_COLS + [TARGET]]

print("\nTARGET        :", TARGET)
print("LEAKAGE_COLS  :", LEAKAGE_COLS)
print("ID_COLS       :", ID_COLS)
print("CATEGORICAL   :", CATEGORICAL)

# This script handles a BINARY target (failure vs no failure)
if df[TARGET].nunique() != 2:
    raise ValueError(f"Target '{TARGET}' has {df[TARGET].nunique()} classes: "
                     f"{df[TARGET].unique()[:10]}. Use a binary failure column as TARGET.")
# Text targets (e.g. "Yes"/"No") -> 0/1
if not pd.api.types.is_numeric_dtype(df[TARGET]):
    mapping = {v: i for i, v in enumerate(sorted(df[TARGET].unique()))}
    print("Target encoding:", mapping, "(make sure 1 = failure!)")
    df[TARGET] = df[TARGET].map(mapping)

# %% 4. EDA
# 4.1 Basic info
df.info()
print("\n--- Descriptive statistics ---")
print(df.describe().T)

# 4.2 Missing values & duplicates
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 4.3 Target distribution (class imbalance check)
counts = df[TARGET].value_counts()
print("\n--- Target distribution ---")
print(counts)
print((counts / len(df) * 100).round(2).astype(str) + " %")

plt.figure(figsize=(5, 4))
sns.countplot(x=TARGET, data=df)
plt.title("Target distribution (0 = OK, 1 = Failure)")
plt.show()

# 4.4 Failure modes breakdown (only for understanding, NOT used as features)
present_leak = [c for c in LEAKAGE_COLS if c in df.columns]
if present_leak:
    print("\n--- Failure mode counts ---")
    print(df[present_leak].sum())

# 4.5 Numerical feature distributions
num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in [TARGET] + LEAKAGE_COLS + ID_COLS]
df[num_cols].hist(bins=30, figsize=(14, 8), edgecolor="black")
plt.suptitle("Numerical feature distributions")
plt.tight_layout()
plt.show()

# 4.6 Outlier check with boxplots (split by target)
fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 4))
for ax, col in zip(np.atleast_1d(axes), num_cols):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# 4.7 Categorical feature vs target
for col in CATEGORICAL:
    print(f"\nFailure rate by {col}:")
    print(df.groupby(col)[TARGET].mean().round(4))
    sns.countplot(x=col, hue=TARGET, data=df)
    plt.title(f"{col} vs {TARGET}")
    plt.show()

# 4.8 Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()

# 4.9 Pairplot (sampled for speed)
sns.pairplot(df[num_cols[:6] + [TARGET]].sample(min(1500, len(df)), random_state=RANDOM_STATE),
             hue=TARGET, corner=True, plot_kws={"alpha": 0.5})
plt.show()

# %% 5. PREPROCESSING
data = df.copy()

# 5.1 Drop duplicates
data = data.drop_duplicates()

# 5.2 Drop identifiers and leakage columns
data = data.drop(columns=[c for c in ID_COLS + LEAKAGE_COLS if c in data.columns])

# 5.3 Clean column names (brackets/spaces can break some libraries)
data.columns = [clean_name(c) for c in data.columns]
TARGET_CLEAN = clean_name(TARGET)
CATEGORICAL = [clean_name(c) for c in CATEGORICAL]
print("Cleaned columns:", list(data.columns))

# 5.4 Feature engineering (physically meaningful features)
if {"Process_temperature", "Air_temperature"}.issubset(data.columns):
    data["Temp_diff"] = data["Process_temperature"] - data["Air_temperature"]
if {"Torque", "Rotational_speed"}.issubset(data.columns):
    # mechanical power (W) = torque * angular velocity
    data["Power"] = data["Torque"] * data["Rotational_speed"] * 2 * np.pi / 60
if {"Tool_wear", "Torque"}.issubset(data.columns):
    data["Wear_x_Torque"] = data["Tool_wear"] * data["Torque"]

# 5.5 Handle missing values (this dataset has none, but kept for general use)
num_features = data.drop(columns=[TARGET_CLEAN]).select_dtypes(include=np.number).columns.tolist()
cat_features = [c for c in data.columns if c in CATEGORICAL]
data[num_features] = data[num_features].fillna(data[num_features].median())
for c in cat_features:
    data[c] = data[c].fillna(data[c].mode()[0])

# 5.6 Split features / target
X = data.drop(columns=[TARGET_CLEAN])
y = data[TARGET_CLEAN]

# 5.7 Stratified train/test split (keeps the rare failure ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Failure rate  train: %.3f  test: %.3f" % (y_train.mean(), y_test.mean()))

# 5.8 Encoding + scaling via ColumnTransformer
# Scaling is REQUIRED for SVM and harmless for RF. Fitting happens inside the
# pipeline on training folds only -> no data leakage.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
])

# NOTE ON IMBALANCE: failures are rare (~3%). Accuracy is misleading, so we use
# class_weight="balanced" and optimise F1 / recall / PR-AUC instead.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %% 6. HELPER: EVALUATION FUNCTION
def evaluate(name, model, X_te, y_te):
    """Print metrics and draw confusion matrix for a fitted pipeline."""
    y_pred = model.predict(X_te)
    # continuous score for ROC / PR curves
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_te)[:, 1]
    else:
        score = model.decision_function(X_te)

    res = {
        "Model": name,
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "ROC-AUC": roc_auc_score(y_te, score),
        "PR-AUC": average_precision_score(y_te, score),
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_te, y_pred, target_names=["No failure", "Failure"]))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=["No failure", "Failure"]).plot(cmap="Blues")
    plt.title(f"{name} - Confusion matrix")
    plt.show()
    return res, score

# %% 7. RANDOM FOREST
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
])

# 7.1 Baseline cross-validation
rf_base = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print("RF baseline CV F1: %.3f +/- %.3f" % (rf_base.mean(), rf_base.std()))

# 7.2 Hyperparameter tuning
rf_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", "log2"],
}
rf_search = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
print("Best RF CV F1: %.3f" % rf_search.best_score_)
best_rf = rf_search.best_estimator_

# 7.3 Test evaluation
rf_res, rf_score = evaluate("Random Forest", best_rf, X_test, y_test)

# 7.4 Feature importance
feat_names = best_rf.named_steps["prep"].get_feature_names_out()
importances = best_rf.named_steps["clf"].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values()
imp.plot(kind="barh", figsize=(8, 5), color="teal")
plt.title("Random Forest - Feature importance")
plt.tight_layout()
plt.show()

# %% 8. SUPPORT VECTOR MACHINE
svm_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", SVC(class_weight="balanced", random_state=RANDOM_STATE)),
])

# 8.1 Baseline cross-validation
svm_base = cross_val_score(svm_pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print("SVM baseline CV F1: %.3f +/- %.3f" % (svm_base.mean(), svm_base.std()))

# 8.2 Hyperparameter tuning
svm_grid = [
    {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
]
svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
svm_search.fit(X_train, y_train)
print("Best SVM params:", svm_search.best_params_)
print("Best SVM CV F1: %.3f" % svm_search.best_score_)
best_svm = svm_search.best_estimator_

# 8.3 Test evaluation
svm_res, svm_score = evaluate("SVM", best_svm, X_test, y_test)

# %% 8B. K-NEAREST NEIGHBORS
# KNN is distance based, so scaling (already in the preprocessor) is essential.
# KNeighborsClassifier has NO class_weight option, so with rare failures it tends
# to predict "no failure". We use weights="distance" and tune on F1 to compensate;
# SMOTE (see last cell) is another good option.
knn_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", KNeighborsClassifier(n_jobs=-1)),
])

# Baseline cross-validation
knn_base = cross_val_score(knn_pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print("KNN baseline CV F1: %.3f +/- %.3f" % (knn_base.mean(), knn_base.std()))

# Hyperparameter tuning
knn_grid = {
    "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"],
}
knn_search = GridSearchCV(knn_pipe, knn_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
knn_search.fit(X_train, y_train)
print("Best KNN params:", knn_search.best_params_)
print("Best KNN CV F1: %.3f" % knn_search.best_score_)
best_knn = knn_search.best_estimator_

# Plot: effect of k on CV F1
k_scores = (pd.DataFrame(knn_search.cv_results_)
            .groupby("param_clf__n_neighbors")["mean_test_score"].max())
plt.figure(figsize=(6, 4))
plt.plot(k_scores.index.astype(int), k_scores.values, marker="o")
plt.xlabel("k (n_neighbors)"); plt.ylabel("Best CV F1")
plt.title("KNN - effect of k"); plt.show()

# Test evaluation
knn_res, knn_score = evaluate("KNN", best_knn, X_test, y_test)

# %% 8C. DECISION TREE
# Trees do not need scaling, but reusing the same preprocessor is harmless
# and keeps the pipeline consistent.
dt_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

# Baseline cross-validation (an unpruned tree usually overfits)
dt_base = cross_val_score(dt_pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print("Decision Tree baseline CV F1: %.3f +/- %.3f" % (dt_base.mean(), dt_base.std()))

# Hyperparameter tuning (depth / leaf size / ccp_alpha control overfitting)
dt_grid = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 5, 7, 10, 15, None],
    "clf__min_samples_split": [2, 10, 20],
    "clf__min_samples_leaf": [1, 5, 10],
    "clf__ccp_alpha": [0.0, 0.0005, 0.001],
}
dt_search = GridSearchCV(dt_pipe, dt_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
dt_search.fit(X_train, y_train)
print("Best DT params:", dt_search.best_params_)
print("Best DT CV F1: %.3f" % dt_search.best_score_)
best_dt = dt_search.best_estimator_

# Test evaluation
dt_res, dt_score = evaluate("Decision Tree", best_dt, X_test, y_test)

# Feature importance
dt_feat_names = best_dt.named_steps["prep"].get_feature_names_out()
dt_imp = pd.Series(best_dt.named_steps["clf"].feature_importances_,
                   index=dt_feat_names).sort_values()
dt_imp.plot(kind="barh", figsize=(8, 5), color="darkorange")
plt.title("Decision Tree - Feature importance")
plt.tight_layout(); plt.show()

# Visualise the tree (first 3 levels so it stays readable)
plt.figure(figsize=(20, 9))
plot_tree(best_dt.named_steps["clf"], feature_names=list(dt_feat_names),
          class_names=["No failure", "Failure"], filled=True, rounded=True,
          max_depth=3, fontsize=8)
plt.title("Decision Tree (top 3 levels)")
plt.show()

# %% 9. MODEL COMPARISON
results = pd.DataFrame([rf_res, svm_res, knn_res, dt_res]).set_index("Model").round(4)
print("\n===== FINAL COMPARISON (test set) =====")
print(results)
results.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))
plt.title("Model comparison")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()

ALL_SCORES = [("Random Forest", rf_score), ("SVM", svm_score),
              ("KNN", knn_score), ("Decision Tree", dt_score)]

# ROC curves
plt.figure(figsize=(6, 5))
for name, sc in ALL_SCORES:
    fpr, tpr, _ = roc_curve(y_test, sc)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, sc):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(); plt.show()

# Precision-Recall curves (better for imbalanced data)
plt.figure(figsize=(6, 5))
for name, sc in ALL_SCORES:
    p, r, _ = precision_recall_curve(y_test, sc)
    plt.plot(r, p, label=f"{name} (AP={average_precision_score(y_test, sc):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curves"); plt.legend(); plt.show()

# %% 10. SAVE BEST MODELS (optional)
# import joblib
# joblib.dump(best_rf, "rf_predictive_maintenance.pkl")
# joblib.dump(best_svm, "svm_predictive_maintenance.pkl")
# joblib.dump(best_knn, "knn_predictive_maintenance.pkl")
# joblib.dump(best_dt, "dt_predictive_maintenance.pkl")

# %% 11. OPTIONAL: SMOTE ALTERNATIVE FOR IMBALANCE
# pip install imbalanced-learn
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import SMOTE
# rf_smote = ImbPipeline([
#     ("prep", preprocessor),
#     ("smote", SMOTE(random_state=RANDOM_STATE)),
#     ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
# ])
# Use imblearn's Pipeline so SMOTE is applied only on training folds.
