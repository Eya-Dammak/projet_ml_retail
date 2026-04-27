"""
predict.py
Inférence sur X_test et sur un client fictif.
Style fonctionnel avec nommage distinct et logs [INFO].
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import charger_train_test, sauvegarder_modele

# Colonnes à exclure (leakage résiduel éventuel)
COLS_A_EXCLURE = ['ChurnRiskCategory', 'CustomerType_Perdu']


# ──────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ──────────────────────────────────────────────────────────────
def charger_tous_les_modeles() -> tuple:
    """
    Charge l'ensemble des artefacts entraînés :
    scaler, RF, XGBoost, Stacking, KMeans, PCA,
    modèle de régression et son scaler dédié.
    """
    scaler      = joblib.load('models/scaler.pkl')
    rf          = joblib.load('models/random_forest.pkl')
    xgb         = joblib.load('models/xgboost.pkl')
    stacking    = joblib.load('models/stacking.pkl')
    kmeans      = joblib.load('models/kmeans.pkl')
    pca         = joblib.load('models/pca.pkl')
    reg_modele  = joblib.load('models/regression_xgboost_optimized.pkl')
    reg_scaler  = joblib.load('models/scaler_regression.pkl')
    print("[INFO] Artefacts chargés : scaler, rf, xgb, stacking, kmeans, pca, "
          "regression_xgboost, scaler_regression")
    return scaler, rf, xgb, stacking, kmeans, pca, reg_modele, reg_scaler


# ──────────────────────────────────────────────────────────────
# PRÉDICTION CHURN — Random Forest
# ──────────────────────────────────────────────────────────────
def predire_churn(client_df: pd.DataFrame, rf) -> dict:
    """
    Applique le Random Forest sur `client_df` après suppression des colonnes
    à exclure et re-standardisation sur X_train.
    Retourne un dict avec la prédiction, le label et les probabilités.
    """
    X_train = pd.read_csv('data/train_test/X_train.csv')
    X_train = X_train.drop(columns=COLS_A_EXCLURE, errors='ignore')
    client  = client_df.drop(columns=COLS_A_EXCLURE, errors='ignore')

    scaler_local = StandardScaler()
    scaler_local.fit(X_train)
    client_sc = scaler_local.transform(client)

    prediction  = rf.predict(client_sc)[0]
    probabilite = rf.predict_proba(client_sc)[0]

    return {
        'churn_predit':  int(prediction),
        'label':         'Churner ⚠️' if prediction == 1 else 'Fidèle ✅',
        'prob_fidele':   round(float(probabilite[0]) * 100, 1),
        'prob_churner':  round(float(probabilite[1]) * 100, 1),
    }


# ──────────────────────────────────────────────────────────────
# PRÉDICTION SEGMENT — K-Means (via PCA)
# ──────────────────────────────────────────────────────────────
def predire_segment(client_df: pd.DataFrame, scaler, pca, kmeans) -> dict:
    """
    Projette le client dans l'espace ACP puis prédit son segment K-Means.
    Retourne un dict avec l'identifiant et le libellé du segment.
    """
    LABELS_SEGMENTS = {
        0: 'Segment A — Clients Premium',
        1: 'Segment B — Clients Réguliers',
        2: 'Segment C — Clients Occasionnels',
        3: 'Segment D — Clients Inactifs',
    }

    client_sc  = scaler.transform(client_df)
    client_pca = pca.transform(client_sc)
    segment    = int(kmeans.predict(client_pca)[0])

    return {
        'segment_id':    segment,
        'segment_label': LABELS_SEGMENTS.get(segment, f'Segment {segment}'),
    }


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLET — Prédictions sur X_test
# ──────────────────────────────────────────────────────────────
def evaluer_sur_test() -> pd.DataFrame:
    """
    Charge X_test, applique le Random Forest et génère un rapport CSV
    avec les prédictions et probabilités pour chaque client du test set.
    """
    print("\n" + "=" * 55)
    print("   PRÉDICTIONS SUR X_TEST (Random Forest)")
    print("=" * 55)

    X_test  = pd.read_csv('data/train_test/X_test.csv')
    y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()
    X_train = pd.read_csv('data/train_test/X_train.csv')

    scaler, rf, xgb, stacking, kmeans, pca, reg_modele, reg_scaler = charger_tous_les_modeles()

    # Nettoyage
    X_test_clf  = X_test.drop(columns=COLS_A_EXCLURE,  errors='ignore')
    X_train_clf = X_train.drop(columns=COLS_A_EXCLURE, errors='ignore')

    scaler_local = StandardScaler()
    scaler_local.fit(X_train_clf)
    X_test_sc = scaler_local.transform(X_test_clf)

    predictions  = rf.predict(X_test_sc)
    probabilites = rf.predict_proba(X_test_sc)

    rapport = pd.DataFrame({
        'Churn_Réel':    y_test.values,
        'Churn_Prédit':  predictions,
        'Prob_Fidèle':   (probabilites[:, 0] * 100).round(1),
        'Prob_Churner':  (probabilites[:, 1] * 100).round(1),
    })

    print(f"\n📊 Aperçu des 10 premières prédictions :")
    print(rapport.head(10).to_string(index=False))

    nb_corrects = (rapport['Churn_Réel'] == rapport['Churn_Prédit']).sum()
    total = len(rapport)
    print(f"\n[INFO] Prédictions correctes : {nb_corrects}/{total} "
          f"({nb_corrects / total * 100:.1f} %)")

    os.makedirs('reports', exist_ok=True)
    rapport.to_csv('reports/predictions_test.csv', index=False)
    print("[INFO] Rapport sauvegardé : reports/predictions_test.csv")
    return rapport


# ──────────────────────────────────────────────────────────────
# EXEMPLE — Client fictif (valeurs moyennes du train)
# ──────────────────────────────────────────────────────────────
def demo_client_fictif() -> None:
    """
    Construit un client fictif (moyennes de X_train) et prédit
    son risque de churn et son segment de clientèle.
    """
    print("\n" + "=" * 55)
    print("   DÉMO — CLIENT FICTIF (valeurs moyennes)")
    print("=" * 55)

    X_train = pd.read_csv('data/train_test/X_train.csv')
    client  = pd.DataFrame([X_train.mean()], columns=X_train.columns)

    scaler, rf, xgb, stacking, kmeans, pca, reg_modele, reg_scaler = charger_tous_les_modeles()

    # Churn
    res_churn   = predire_churn(client.copy(), rf)
    res_segment = predire_segment(client.copy(), scaler, pca, kmeans)

    print(f"\n👤 Profil client fictif (moyennes du train) :")
    print(f"   Risque churn  : {res_churn['label']}")
    print(f"   Prob. Fidèle  : {res_churn['prob_fidele']} %")
    print(f"   Prob. Churner : {res_churn['prob_churner']} %")
    print(f"   Segment       : {res_segment['segment_label']}")


# ──────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluer_sur_test()
    demo_client_fictif()
    print("\n[DONE] predict.py terminé avec succès !")


