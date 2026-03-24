# ============================================================
# 🚀 APP.PY - VERSION PARFAITE (FIX SCALING)
# ============================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ============================================================
# 📦 LOAD MODELS
# ============================================================

# clustering
kmeans = joblib.load("../models/kmeans.pkl")
pca = joblib.load("../models/pca.pkl")
scaler_cluster = joblib.load("../models/scaler_cluster.pkl")

cluster_features = joblib.load("../models/cluster_features.pkl")

# classification
clf = joblib.load("../models/churn_model.pkl")
clf_columns = joblib.load("../models/churn_columns.pkl")
scaler_clf = joblib.load("../models/scaler_clf.pkl")  # 🔥 FIX

# regression
reg = joblib.load("../models/regression_model.pkl")
reg_columns = joblib.load("../models/reg_columns.pkl")
scaler_reg = joblib.load("../models/scaler_reg.pkl")  # 🔥 FIX

print("✅ Modèles chargés")

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

# ============================================================
# 🏠 ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        try:
            # =================================================
            # 📥 INPUT
            # =================================================
            data = {
                "Recency": float(request.form["Recency"]),
                "Frequency": float(request.form["Frequency"]),
                "MonetaryTotal": float(request.form["MonetaryTotal"]),
                "CustomerTenureDays": float(request.form["CustomerTenure"]),
                "AvgDaysBetweenPurchases": float(request.form["AvgDaysBetween"]),
                "TotalTransactions": float(request.form["TotalTrans"])
            }

            df = pd.DataFrame([data])

            # =================================================
            # 🔵 CLUSTERING
            # =================================================
            df_cluster = df.reindex(columns=cluster_features, fill_value=0)
            df_cluster = df_cluster.astype(float)

            X_scaled = scaler_cluster.transform(df_cluster)
            X_pca = pca.transform(X_scaled)

            cluster = kmeans.predict(X_pca)[0]
            interpretation = interpret_cluster(cluster)

            # =================================================
            # 🟢 CHURN (🔥 FIX SCALING)
            # =================================================
            df_clf = df.copy()
            df_clf["Cluster"] = cluster

            df_clf = pd.get_dummies(df_clf)
            df_clf = df_clf.reindex(columns=clf_columns, fill_value=0)

            X_clf_scaled = scaler_clf.transform(df_clf)  # 🔥 CRITIQUE

            churn_pred = clf.predict(X_clf_scaled)[0]
            churn_proba = clf.predict_proba(X_clf_scaled)[0][1]

            # =================================================
            # 🟣 REGRESSION (🔥 FIX SCALING)
            # =================================================
            df_reg = df.copy()
            df_reg["Cluster"] = cluster

            df_reg = pd.get_dummies(df_reg)
            df_reg = df_reg.reindex(columns=reg_columns, fill_value=0)

            X_reg_scaled = scaler_reg.transform(df_reg)  # 🔥 CRITIQUE

            revenue = reg.predict(X_reg_scaled)[0]

            # =================================================
            # 📤 RESULT
            # =================================================
            return render_template(
                "index.html",
                interpretation=interpretation,   # ❌ on enlève cluster
                churn=int(churn_pred),           # ✅ force 0 ou 1
                proba=round(churn_proba, 2),
                revenue=round(revenue, 2)
            )

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

# ============================================================
# ▶️ RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)