"""
CUSTOMER CHURN PREDICTION PIPELINE
===================================
Works with telecom / banking / SaaS churn datasets (e.g. the popular
"Telco Customer Churn" dataset from Kaggle: 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Just point DATA_PATH at your CSV and set TARGET_COL to your churn column name.
Expected target column values: Yes/No, 1/0, or True/False (auto-handled below).

Author: Generated for Abhijeet's B.Tech project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not installed. Run: pip install xgboost")

# -------------------------------------------------------------------
# 1. CONFIG — change these two lines for your dataset
# -------------------------------------------------------------------
DATA_PATH = "telco_churn.csv"     # path to your CSV
TARGET_COL = "Churn"              # name of the churn column
CUSTOMER_ID_COL = "customerID"    # column to drop (unique identifier), or None

# -------------------------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# -------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------
def run_eda(df, target_col):
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n--- Churn Distribution ---")
    print(df[target_col].value_counts(normalize=True) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Churn distribution
    df[target_col].value_counts().plot(kind="bar", ax=axes[0, 0], color=["#4C72B0", "#DD8452"])
    axes[0, 0].set_title("Churn Distribution")

    # Numeric feature correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 1])
        axes[0, 1].set_title("Correlation Heatmap (numeric features)")

    # Tenure vs churn (if exists)
    if "tenure" in df.columns:
        sns.histplot(data=df, x="tenure", hue=target_col, multiple="stack", ax=axes[1, 0])
        axes[1, 0].set_title("Tenure vs Churn")

    # Monthly charges vs churn (if exists)
    if "MonthlyCharges" in df.columns:
        sns.boxplot(data=df, x=target_col, y="MonthlyCharges", ax=axes[1, 1])
        axes[1, 1].set_title("Monthly Charges vs Churn")

    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=150)
    print("\nSaved EDA plots to eda_overview.png")
    plt.close()

# -------------------------------------------------------------------
# 4. PREPROCESSING
# -------------------------------------------------------------------
def preprocess(df, target_col, id_col=None):
    df = df.copy()

    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])

    # Normalize target to 0/1 (robust to pandas string/object/Arrow dtypes)
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .map({"Yes": 1, "No": 0, "True": 1, "False": 0, "1": 1, "0": 0})
    )
    if df[target_col].isnull().any():
        raise ValueError(
            f"Unrecognized values in '{target_col}' column after mapping. "
            f"Check for typos or unexpected categories."
        )
    df[target_col] = df[target_col].astype(int)

    # TotalCharges sometimes loads as string with blanks (common Telco dataset quirk)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Fill remaining missing values
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns (catches 'object' AND newer pandas 'str'/Arrow-backed dtypes)
    cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y

# -------------------------------------------------------------------
# 5. MODEL TRAINING
# -------------------------------------------------------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", scale_pos_weight=1, random_state=42
        )

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained: {name}")

    return models

# -------------------------------------------------------------------
# 6. EVALUATION
# -------------------------------------------------------------------
def evaluate_models(models, X_test, y_test):
    results = []
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
        })

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.savefig("roc_comparison.png", dpi=150)
    plt.close()
    print("\nSaved ROC comparison to roc_comparison.png")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print("\n=== Model Comparison Summary ===")
    print(results_df.to_string(index=False))
    return results_df

# -------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Random Forest / XGBoost)
# -------------------------------------------------------------------
def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title(f"Top 15 Feature Importances — {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fname = f"feature_importance_{model_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {fname}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    run_eda(df, TARGET_COL)

    X, y = preprocess(df, TARGET_COL, CUSTOMER_ID_COL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)

    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"], X.columns, "Random Forest")
    if "XGBoost" in models:
        plot_feature_importance(models["XGBoost"], X.columns, "XGBoost")

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nDone. Results saved to model_comparison_results.csv")