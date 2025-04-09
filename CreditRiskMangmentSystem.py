# ============================
# 🗓️ DAY 1: Initial Setup, Utilities & Data Loading
# ============================
# !pip install catboost
# !pip install --upgrade numpy
# !pip install --upgrade catboost
# !pip install --upgrade numpy  # Upgrade numpy to the latest version first
# !pip install --force-reinstall catboost  # Force reinstall CatBoost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap
import warnings
import time
import os

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ✅ Utility function to track time usage
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"\n⏱️ Time taken by '{func.__name__}': {end - start:.2f}s")
        return result
    return wrapper

# ✅ Print data info, columns, and descriptive stats
def describe_data(df):
    print("\n🔍 Data Info:")
    print(df.info())
    print("\n🧾 Data Description:")
    print(df.describe(include='all'))
    print("\n🆔 Column Names:", df.columns.tolist())

# ✅ Load CSV file into DataFrame
@timer
def load_data(path):
    df = pd.read_csv(path)
    print("\n✅ Data loaded successfully! Shape:", df.shape)
    return df


# ============================
# 🗓️ DAY 2: Exploratory Data Analysis
# ============================

@timer
def run_eda(df):
    describe_data(df)
    print("\n📊 Missing values:\n", df.isnull().sum())
    print("\n📈 Class Distribution:\n", df[df.columns[-1]].value_counts())

    plt.figure(figsize=(8, 4))
    sns.countplot(x=df.columns[-1], data=df)
    plt.title("Target Variable Distribution")
    plt.show()

    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_features) > 1:
        plt.figure(figsize=(14, 6))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

    for col in numerical_features:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()


# ============================
# 🗓️ DAY 3: Feature Engineering & Preprocessing
# ============================

@timer
def feature_engineering(df):
    df = df.copy()
    if 'Income' in df.columns and 'LoanAmount' in df.columns:
        df['Income_to_Loan_Ratio'] = df['Income'] / (df['LoanAmount'] + 1)
    if 'Age' in df.columns:
        df['Age_Bin'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], labels=["Young", "Mid-Age", "Senior", "Elder"])
    return df

@timer
def preprocess(df, target):
    df = df.dropna(subset=[target]) # Drop rows with NaN in the target column before splitting.
    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    return X, y, preprocessor


# ============================
# 🗓️ DAY 4: Model Training and Evaluation
# ============================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

@timer
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n🚀 Training Model: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("✔ Accuracy:", accuracy_score(y_test, y_pred))
    print("🎯 F1 Score:", f1_score(y_test, y_pred))
    print("🔥 ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")


# ============================
# 🗓️ DAY 5: Model Explainability & Integration
# ============================

@timer
def explain_model(model, X_train):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train[:100])
        shap.summary_plot(shap_values, X_train[:100])
    except Exception as e:
        print("❌ SHAP Explainability Failed:", e)

@timer
def main():
    FILE_PATH = "train1.csv"  # 🔁 Replace with your dataset
    TARGET = "credit_card_default"                   # 🎯 Replace with your actual target column

    df = load_data(FILE_PATH)
    run_eda(df)
    df = feature_engineering(df)

    X, y, preprocessor = preprocess(df, TARGET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    models = get_models()

    plt.figure(figsize=(10, 6))
    for model_name, model_obj in models.items():
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('oversample', smote),
            ('classifier', model_obj)
        ])
        evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Model ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    best_model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('oversample', smote),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    best_model.fit(X_train, y_train)
    transformed = preprocessor.fit_transform(X_train)
    explain_model(best_model.named_steps['classifier'], pd.DataFrame(transformed))

# Start the 5-day workflow
if __name__ == '__main__':
    main()
