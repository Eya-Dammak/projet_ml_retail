# ============================================================
# 📊 AFFICHER LES CLUSTERS + INTERVALLES
# ============================================================

import pandas as pd

# ============================================================
# 📥 LOAD DATA AVEC CLUSTERS
# ============================================================

df = pd.read_csv("../data/processed/customers_segmented.csv")

print("📊 Dataset chargé :", df.shape)

# ============================================================
# 🔥 FEATURES UTILISÉES POUR CLUSTERING
# ============================================================

features = [
    "Recency",
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "AvgDaysBetweenPurchases",
    "TotalTransactions"
]

# ============================================================
# 📊 CALCUL DES INTERVALLES
# ============================================================

cluster_summary = df.groupby("Cluster")[features].agg(["mean", "min", "max"])

# ============================================================
# 🎯 AFFICHAGE PROPRE
# ============================================================

for cluster_id in cluster_summary.index:

    print("\n" + "="*50)
    print(f"🧠 CLUSTER {cluster_id}")
    print("="*50)

    for feature in features:

        mean_val = cluster_summary.loc[cluster_id, (feature, "mean")]
        min_val = cluster_summary.loc[cluster_id, (feature, "min")]
        max_val = cluster_summary.loc[cluster_id, (feature, "max")]

        print(f"{feature}:")
        print(f"   ➤ Moyenne : {round(mean_val, 2)}")
        print(f"   ➤ Intervalle : [{round(min_val, 2)} → {round(max_val, 2)}]")
        print()

# ============================================================
# 💾 SAUVEGARDE CSV PROPRE
# ============================================================

# Reformater pour CSV simple
flat_summary = df.groupby("Cluster")[features].mean()

for col in features:
    flat_summary[col + "_min"] = df.groupby("Cluster")[col].min()
    flat_summary[col + "_max"] = df.groupby("Cluster")[col].max()

flat_summary.to_csv("../reports/cluster_intervals_clean.csv")

print("\n✅ Fichier sauvegardé : reports/cluster_intervals_clean.csv")