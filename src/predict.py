# ============================================================
# 🔮 PREDICT.PY - VERSION FINALE PARFAITE
# Clustering + Churn + Revenue (AVEC SCALING)
# ============================================================

import pandas as pd
import joblib

# ============================================================
# 📦 LOAD MODELS
# ============================================================

# 🔵 Clustering
kmeans = joblib.load("../models/kmeans.pkl")
pca = joblib.load("../models/pca.pkl")
scaler_cluster = joblib.load("../models/scaler_cluster.pkl")
cluster_features = joblib.load("../models/cluster_features.pkl")

# 🟢 Classification
clf = joblib.load("../models/churn_model.pkl")
clf_columns = joblib.load("../models/churn_columns.pkl")
scaler_clf = joblib.load("../models/scaler_clf.pkl")  # 🔥 FIX

# 🟣 Regression
reg = joblib.load("../models/regression_model.pkl")
reg_columns = joblib.load("../models/reg_columns.pkl")
scaler_reg = joblib.load("../models/scaler_reg.pkl")  # 🔥 FIX

print("✅ Modèles chargés")

# ============================================================
# 👤 NOUVEAU CLIENT (TEST)
# ============================================================

new_client = pd.DataFrame([{
    "Recency": 10,
    "Frequency": 5,
    "MonetaryTotal": 200,
    "CustomerTenureDays": 300,
    "AvgDaysBetweenPurchases": 20,
    "TotalTransactions": 15
}])

print("\n👤 Client testé :")
print(new_client)

# ============================================================
# 🔵 1. CLUSTERING
# ============================================================

# 🔥 alignement avec train_model
df_cluster = new_client.reindex(columns=cluster_features, fill_value=0)
df_cluster = df_cluster.astype(float)

# scaling + PCA
X_scaled = scaler_cluster.transform(df_cluster)
X_pca = pca.transform(X_scaled)

cluster = kmeans.predict(X_pca)[0]

print(f"\n🎯 Cluster prédit : {cluster}")

# ============================================================
# 🧠 INTERPRÉTATION
# ============================================================

def interpret_cluster(cluster_id):
    return {
        0: "🟡 Clients occasionnels",
        1: "🔴 Clients à risque",
        2: "💰 Clients VIP",
        3: "🟡 Clients occasionnels",
        4: "💰 Gros acheteurs",
        5: "🟡 Clients peu actifs",
        6: "💰 Clients fidèles"
    }.get(cluster_id, "Cluster inconnu")

print("📊 Interprétation :", interpret_cluster(cluster))

# ============================================================
# 🟢 2. CHURN (AVEC SCALING)
# ============================================================

df_clf = new_client.copy()
df_clf["Cluster"] = cluster

# encoding + alignement
df_clf = pd.get_dummies(df_clf)
df_clf = df_clf.reindex(columns=clf_columns, fill_value=0)

# 🔥 scaling obligatoire
X_clf_scaled = scaler_clf.transform(df_clf)

churn_pred = clf.predict(X_clf_scaled)[0]
churn_proba = clf.predict_proba(X_clf_scaled)[0][1]

print(f"\n📉 Churn (0=stable, 1=risque) : {churn_pred}")
print(f"📊 Probabilité de churn : {churn_proba:.2f}")

# ============================================================
# 🟣 3. REGRESSION (AVEC SCALING)
# ============================================================

df_reg = new_client.copy()
df_reg["Cluster"] = cluster

# encoding + alignement
df_reg = pd.get_dummies(df_reg)
df_reg = df_reg.reindex(columns=reg_columns, fill_value=0)

# 🔥 scaling obligatoire
X_reg_scaled = scaler_reg.transform(df_reg)

revenue_pred = reg.predict(X_reg_scaled)[0]

print(f"\n💰 Revenu prédit : {revenue_pred:.2f} £")

# ============================================================
# ✅ FIN
# ============================================================

print("\n🚀 Test terminé avec succès !")