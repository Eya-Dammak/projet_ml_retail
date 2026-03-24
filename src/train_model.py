# ============================================================
# 🚀 TRAIN_MODEL.PY - VERSION CORRIGÉE (FIABLE)
# ============================================================

import pandas as pd
import numpy as np
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

print("\n🔵 CLUSTERING + PCA")

features_cluster = [
    "Recency",
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "AvgDaysBetweenPurchases",
    "TotalTransactions"
]

# Vérification
missing = [f for f in features_cluster if f not in df.columns]
if missing:
    raise ValueError(f"❌ Colonnes manquantes : {missing}")

df_cluster = df[features_cluster].copy()
df_cluster = df_cluster.fillna(df_cluster.mean())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Nb composantes PCA :", pca.n_components_)

# Choix meilleur K
best_k = 2
best_score = -1

for k in range(2, 10):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)

    print(f"k={k}, silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"🔥 Meilleur K : {best_k}")

# Clustering final
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_pca)

# ============================================================
# 🟢 CLASSIFICATION (CHURN)
# ============================================================

print("\n🟢 CLASSIFICATION (CHURN)")

df_clf = df.copy()

# ❌ SUPPRIMER CLUSTER (évite data leakage)
df_clf = df_clf.drop("Cluster", axis=1)

df_clf = pd.get_dummies(df_clf, drop_first=True)

X = df_clf.drop("Churn", axis=1)
y = df_clf["Churn"]

# 🔥 Scaling
scaler_clf = StandardScaler()
X_scaled = scaler_clf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

def objective_clf(trial):

    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        class_weight="balanced",
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    return score.mean()

study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(objective_clf, n_trials=20)

print("🔥 Best params (Churn):", study_clf.best_params)

clf = RandomForestClassifier(**study_clf.best_params,
                             class_weight="balanced",
                             random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n📊 Classification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print("ROC AUC:", roc)

# ============================================================
# 🟣 REGRESSION (MonetaryTotal)
# ============================================================

print("\n🟣 REGRESSION (Revenue)")

df_reg = df.copy()
df_reg = df_reg.drop("Cluster", axis=1)

df_reg = pd.get_dummies(df_reg, drop_first=True)

X_reg = df_reg.drop("MonetaryTotal", axis=1)
y_reg = df_reg["MonetaryTotal"]

# Scaling
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

def objective_reg(trial):

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        random_state=42
    )

    score = cross_val_score(
        model,
        X_train_r,
        y_train_r,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    return -score.mean()  # 🔥 correction

study_reg = optuna.create_study(direction="minimize")
study_reg.optimize(objective_reg, n_trials=20)

print("🔥 Best params (Revenue):", study_reg.best_params)

reg = RandomForestRegressor(**study_reg.best_params, random_state=42)
reg.fit(X_train_r, y_train_r)

y_pred_r = reg.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("\n📊 MAE:", mae)
print("📊 R2:", r2)

# ============================================================
# 💾 SAVE MODELS
# ============================================================

print("\n💾 Sauvegarde des modèles...")

os.makedirs("../models", exist_ok=True)

joblib.dump(kmeans, "../models/kmeans.pkl")
joblib.dump(pca, "../models/pca.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(features_cluster, "../models/cluster_features.pkl")

joblib.dump(clf, "../models/churn_model.pkl")
joblib.dump(X.columns, "../models/churn_columns.pkl")

joblib.dump(reg, "../models/regression_model.pkl")
joblib.dump(X_reg.columns, "../models/reg_columns.pkl")

df.to_csv("../data/processed/customers_segmented.csv", index=False)

print("\n🚀 PROJET COMPLET TERMINÉ SANS ERREURS 🔥")
# ============================================================
# 📊 ANALYSE DES CLUSTERS
# ============================================================

print("\n📊 ANALYSE DES CLUSTERS")

# Moyennes par cluster
cluster_means = df.groupby("Cluster")[features_cluster].mean()

# Min / Max par cluster
cluster_min = df.groupby("Cluster")[features_cluster].min()
cluster_max = df.groupby("Cluster")[features_cluster].max()

# Fusionner tout
cluster_summary = cluster_means.copy()

for col in features_cluster:
    cluster_summary[col + "_min"] = cluster_min[col]
    cluster_summary[col + "_max"] = cluster_max[col]

print("\n📊 Résumé des clusters :")
print(cluster_summary)

# Sauvegarde
cluster_summary.to_csv("../reports/cluster_analysis.csv")

print("\n✅ Analyse clusters sauvegardée dans reports/cluster_analysis.csv")
print("\n🧠 INTERPRÉTATION DES CLUSTERS")

for cluster_id in cluster_summary.index:
    recency = cluster_summary.loc[cluster_id, "Recency"]
    frequency = cluster_summary.loc[cluster_id, "Frequency"]
    monetary = cluster_summary.loc[cluster_id, "MonetaryTotal"]

    if frequency > 20 and monetary > 3000:
        label = "💰 Gros acheteurs (VIP)"
    elif recency > 150 and frequency < 5:
        label = "🔴 Clients à risque"
    elif frequency < 10:
        label = "🟡 Clients occasionnels"
    else:
        label = "🟢 Clients fidèles"

    print(f"Cluster {cluster_id} → {label}")