# ============================================================
# 🚀 TRAIN_MODEL FINAL (CORRIGÉ PROPRE)
# ============================================================

import pandas as pd
import os
import joblib
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    r2_score
)

# ============================================================
# 📥 LOAD DATA
# ============================================================

df = pd.read_csv("../data/train_test/train.csv")
print("📊 Dataset chargé :", df.shape)

# ============================================================
# 🔵 CLUSTERING + PCA
# ============================================================

features_cluster = [
    "Recency",
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "AvgDaysBetweenPurchases",
    "TotalTransactions"
]

df_cluster = df[features_cluster].copy()

# garder seulement colonnes numériques (sécurité)
df_cluster = df_cluster.select_dtypes(include=["number"])

df_cluster = df_cluster.fillna(df_cluster.mean())
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(df_cluster)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 🔥 BEST K
best_k, best_score = 2, -1
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    score = silhouette_score(X_pca, km.fit_predict(X_pca))
    if score > best_score:
        best_k, best_score = k, score

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_pca)

print("🔥 Best K =", best_k)

# ============================================================
# 🟢 CLASSIFICATION
# ============================================================

df_clf = df.drop("Cluster", axis=1)
df_clf = pd.get_dummies(df_clf, drop_first=True)

X = df_clf.drop("Churn", axis=1)
y = df_clf["Churn"]

# 🔥 IMPORTANT
scaler_clf = StandardScaler()
X_scaled = scaler_clf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

def objective_clf(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        class_weight="balanced",
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective_clf, n_trials=20)

clf = RandomForestClassifier(**study.best_params, class_weight="balanced")
clf.fit(X_train, y_train)

print("ROC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

# ============================================================
# 🟣 REGRESSION
# ============================================================

df_reg = df.drop("Cluster", axis=1)
df_reg = pd.get_dummies(df_reg, drop_first=True)

X_reg = df_reg.drop("MonetaryTotal", axis=1)
y_reg = df_reg["MonetaryTotal"]

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

def objective_reg(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        random_state=42
    )
    score = cross_val_score(model, X_train_r, y_train_r, cv=5,
                            scoring="neg_mean_absolute_error")
    return -score.mean()

study_reg = optuna.create_study(direction="minimize")
study_reg.optimize(objective_reg, n_trials=20)

reg = RandomForestRegressor(**study_reg.best_params)
reg.fit(X_train_r, y_train_r)

print("MAE:", mean_absolute_error(y_test_r, reg.predict(X_test_r)))

# ============================================================
# 💾 SAVE (TRÈS IMPORTANT)
# ============================================================

os.makedirs("../models", exist_ok=True)

# clustering
joblib.dump(kmeans, "../models/kmeans.pkl")
joblib.dump(pca, "../models/pca.pkl")
joblib.dump(scaler_cluster, "../models/scaler_cluster.pkl")
joblib.dump(features_cluster, "../models/cluster_features.pkl")

# classification
joblib.dump(clf, "../models/churn_model.pkl")
joblib.dump(scaler_clf, "../models/scaler_clf.pkl")   # 🔥 FIX
joblib.dump(X.columns, "../models/churn_columns.pkl")

# regression
joblib.dump(reg, "../models/regression_model.pkl")
joblib.dump(scaler_reg, "../models/scaler_reg.pkl")   # 🔥 FIX
joblib.dump(X_reg.columns, "../models/reg_columns.pkl")

print("\n✅ MODELS SAUVEGARDÉS SANS BUG")