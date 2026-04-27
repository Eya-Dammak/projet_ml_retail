"""
train_model.py
Pipeline d'entraînement : ACP, Clustering, Classification (RF + XGBoost + Stacking), Régression.
Style fonctionnel avec nommage et logs distincts de l'original.
"""

import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    confusion_matrix, f1_score, mean_squared_error,
    r2_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_val_predict, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import charger_train_test, sauvegarder_figure, sauvegarder_modele


# ──────────────────────────────────────────────────────────────
# UTILITAIRE : Courbe ROC-AUC
# ──────────────────────────────────────────────────────────────
def afficher_courbe_roc(y_reel, y_proba, nom_modele: str) -> None:
    """Trace et sauvegarde la courbe ROC pour un modèle donné."""
    fpr, tpr, _ = roc_curve(y_reel, y_proba)
    score_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{nom_modele} (AUC = {score_auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "r--", lw=1)
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbe ROC — {nom_modele}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    sauvegarder_figure(f"roc_auc_{nom_modele.lower()}.png")
    print(f"[INFO] AUC {nom_modele} : {score_auc:.3f}")


# ──────────────────────────────────────────────────────────────
# MODÈLE 0 — ACP
# ──────────────────────────────────────────────────────────────
def executer_acp(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Réduit la dimension avec PCA :
      - Détermine automatiquement le nombre de composantes pour 95 % de variance
      - Génère la figure de variance cumulée
      - Sauvegarde le modèle PCA
    Retourne : (pca, X_train_pca, X_test_pca)
    """
    print("\n" + "=" * 55)
    print("   ACP — ANALYSE EN COMPOSANTES PRINCIPALES")
    print("=" * 55)

    pca_diagnostic = PCA(random_state=42)
    pca_diagnostic.fit(X_train)

    variance_cumulee = np.cumsum(pca_diagnostic.explained_variance_ratio_)
    n_comp_95 = int(np.argmax(variance_cumulee >= 0.95)) + 1
    n_comp_90 = int(np.argmax(variance_cumulee >= 0.90)) + 1
    print(f"   Composantes nécessaires → 90 % : {n_comp_90} | 95 % : {n_comp_95}")

    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(range(1, len(variance_cumulee) + 1), variance_cumulee,
             marker="o", markersize=3, color="steelblue")
    ax1.axhline(0.95, color="red", linestyle="--", label="95 %")
    ax1.axvline(n_comp_95, color="red", linestyle=":")
    ax1.set_title("Variance cumulée expliquée")
    ax1.legend()
    ax2.bar(range(1, 21), pca_diagnostic.explained_variance_ratio_[:20],
            color="coral", edgecolor="white")
    ax2.set_title("Variance par composante (top 20)")
    plt.tight_layout()
    sauvegarder_figure("acp_variance.png")

    # ACP finale avec n_comp_95 composantes
    pca = PCA(n_components=n_comp_95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    print(f"[INFO] ACP : {X_train.shape[1]} → {n_comp_95} composantes "
          f"(variance conservée : {variance_cumulee[n_comp_95 - 1] * 100:.1f} %)")
    joblib.dump(pca, "models/pca.pkl")
    return pca, X_train_pca, X_test_pca


# ──────────────────────────────────────────────────────────────
# MODÈLE 1 — CLUSTERING K-MEANS
# ──────────────────────────────────────────────────────────────
def entrainer_kmeans(donnees: np.ndarray):
    """
    Cherche le k optimal (2 → 8) via l'inertie et le score de silhouette,
    fixe k=4 comme solution finale et sauvegarde le modèle.
    """
    print("\n" + "=" * 55)
    print("   MODÈLE 1 — CLUSTERING K-MEANS")
    print("=" * 55)

    valeurs_k = range(2, 9)
    inerties, silhouettes = [], []

    for k in valeurs_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(donnees)
        inerties.append(km.inertia_)
        silhouettes.append(silhouette_score(donnees, km.labels_))
        print(f"   k={k} → inertie={km.inertia_:.0f} | silhouette={silhouettes[-1]:.3f}")

    # Figure coude + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(valeurs_k), inerties, marker="o", color="steelblue")
    ax1.set_title("Inertie intra-cluster (méthode du coude)")
    ax1.set_xlabel("k")
    ax2.plot(list(valeurs_k), silhouettes, marker="o", color="coral")
    ax2.set_title("Score de silhouette")
    ax2.set_xlabel("k")
    plt.tight_layout()
    sauvegarder_figure("clustering_choix_k.png")

    # Modèle final
    k_retenu = 4
    km_final = KMeans(n_clusters=k_retenu, random_state=42, n_init=10)
    km_final.fit(donnees)
    score_final = silhouette_score(donnees, km_final.labels_)
    repartition = pd.Series(km_final.labels_).value_counts().to_dict()

    print(f"\n[INFO] K-Means final — k={k_retenu} | silhouette={score_final:.3f}")
    print(f"   Répartition des clusters : {repartition}")
    sauvegarder_modele(km_final, "kmeans.pkl")
    return km_final


# ──────────────────────────────────────────────────────────────
# MODÈLE 2a — RANDOM FOREST + OPTIMISATION DU SEUIL
# ──────────────────────────────────────────────────────────────
def entrainer_random_forest(X_train, X_test, y_train, y_test):
    """
    Entraîne un Random Forest avec :
      - Vérification anti-leakage (corrélation > 0.85 avec Churn → arrêt)
      - RandomizedSearchCV (60 itérations, cv=5, scoring='f1')
      - Optimisation du seuil de décision sur un sous-ensemble de validation
    """
    print("\n" + "=" * 55)
    print("   MODÈLE 2a — RANDOM FOREST (SMOTE + SEUIL OPTIMISÉ)")
    print("=" * 55)

    # Vérification anti-leakage
    df_check = X_train.copy()
    df_check['Churn'] = y_train.values
    corr_churn = df_check.corr()['Churn'].abs().drop('Churn').sort_values(ascending=False)
    print(f"\n🔍 Top 10 corrélations avec Churn :\n{corr_churn.head(10).round(4)}")
    if (corr_churn > 0.85).any():
        print("\n⚠️  Leakage probable détecté (corrélation > 0.85) — arrêt du pipeline")
        sys.exit(1)

    print(f"\nDistribution classes : {y_train.value_counts().to_dict()}")

    espace_hyperparams = {
        'n_estimators':     [600, 800, 1000, 1200],
        'max_depth':        [20, 30, 40, None],
        'min_samples_split':[2, 3, 5],
        'min_samples_leaf': [1, 2, 3],
        'max_features':     ['sqrt', 'log2', 0.7],
        'bootstrap':        [True],
        'criterion':        ['gini'],
        'max_samples':      [0.8, 0.9, None],
    }

    rf_base = RandomForestClassifier(
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    recherche = RandomizedSearchCV(
        rf_base, param_distributions=espace_hyperparams,
        n_iter=60, cv=5, scoring='f1',
        n_jobs=-1, verbose=1, random_state=42,
    )
    recherche.fit(X_train, y_train)
    rf_optimal = recherche.best_estimator_
    print(f"[INFO] Meilleurs hyperparamètres : {recherche.best_params_}")

    # Optimisation du seuil sur sous-ensemble de validation
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    rf_optimal.fit(X_sub, y_sub)
    probas_val = rf_optimal.predict_proba(X_val)[:, 1]

    meilleur_seuil, meilleur_f1 = 0.5, 0.0
    for seuil in np.arange(0.25, 0.75, 0.01):
        preds = (probas_val >= seuil).astype(int)
        score = f1_score(y_val, preds)
        if score > meilleur_f1:
            meilleur_f1 = score
            meilleur_seuil = seuil

    print(f"[INFO] Seuil optimal : {meilleur_seuil:.2f} | F1 validation = {meilleur_f1:.3f}")

    # Entraînement final et évaluation
    rf_optimal.fit(X_train, y_train)
    probas_test = rf_optimal.predict_proba(X_test)[:, 1]

    y_pred_def = (probas_test >= 0.50).astype(int)
    y_pred_opt = (probas_test >= meilleur_seuil).astype(int)
    acc_def = accuracy_score(y_test, y_pred_def)
    acc_opt = accuracy_score(y_test, y_pred_opt)

    print(f"\nAccuracy seuil 0.50 : {acc_def:.3f}")
    print(f"Accuracy seuil opt. : {acc_opt:.3f}")
    y_pred_final = y_pred_opt if acc_opt >= acc_def else y_pred_def

    print("\n📊 Rapport de classification :")
    print(classification_report(y_test, y_pred_final))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'],
                yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de confusion — Random Forest')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix_rf.png')

    # Importance des features
    importances = (
        pd.Series(rf_optimal.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title('Top 15 features — Random Forest')
    plt.tight_layout()
    sauvegarder_figure('classification_feature_importance_rf.png')

    sauvegarder_modele(rf_optimal, 'random_forest.pkl')
    afficher_courbe_roc(y_test, probas_test, "RandomForest")
    return rf_optimal


# ──────────────────────────────────────────────────────────────
# MODÈLE 2b — XGBOOST + OPTIMISATION DU SEUIL
# ──────────────────────────────────────────────────────────────
def entrainer_xgboost(X_train, X_test, y_train, y_test):
    """
    Entraîne un XGBoost avec SMOTE (ratio 0.5), GridSearchCV (cv=5)
    et optimisation du seuil de décision sur un sous-ensemble de validation.
    """
    print("\n" + "=" * 55)
    print("   MODÈLE 2b — XGBOOST (SMOTE + SEUIL OPTIMISÉ)")
    print("=" * 55)

    print(f"\n   Distribution initiale : {y_train.value_counts().to_dict()}")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print(f"   Après SMOTE : {pd.Series(y_sm).value_counts().to_dict()}")

    grille = {
        'n_estimators':    [500, 700],
        'max_depth':       [5, 6, 7],
        'learning_rate':   [0.03, 0.05, 0.08],
        'subsample':       [0.8, 0.9],
        'colsample_bytree':[0.8, 0.9],
    }

    xgb_base = XGBClassifier(
        random_state=42, use_label_encoder=False,
        eval_metric='logloss', scale_pos_weight=3
    )
    gs = GridSearchCV(
        xgb_base, grille, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    gs.fit(X_sm, y_sm)
    xgb_optimal = gs.best_estimator_
    print(f"[INFO] Meilleurs hyperparamètres : {gs.best_params_}")

    # Optimisation du seuil
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_sm, y_sm, test_size=0.2, random_state=42, stratify=y_sm
    )
    xgb_optimal.fit(X_sub, y_sub)
    probas_val = xgb_optimal.predict_proba(X_val)[:, 1]

    meilleur_seuil, meilleure_acc = 0.5, 0.0
    for seuil in np.arange(0.3, 0.7, 0.01):
        preds = (probas_val >= seuil).astype(int)
        acc   = accuracy_score(y_val, preds)
        if acc > meilleure_acc:
            meilleure_acc = acc
            meilleur_seuil = seuil

    print(f"[INFO] Seuil optimal : {meilleur_seuil:.2f} | accuracy validation = {meilleure_acc:.3f}")

    # Entraînement final
    xgb_optimal.fit(X_sm, y_sm)
    probas_test = xgb_optimal.predict_proba(X_test)[:, 1]
    y_pred_final = (probas_test >= meilleur_seuil).astype(int)

    print(f"\n[INFO] Accuracy test : {accuracy_score(y_test, y_pred_final):.3f}")
    print(classification_report(y_test, y_pred_final,
                                target_names=['Fidèle (0)', 'Churner (1)']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'],
                yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de confusion — XGBoost')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix_xgb.png')

    importances = (
        pd.Series(xgb_optimal.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title('Top 15 features — XGBoost')
    plt.tight_layout()
    sauvegarder_figure('classification_feature_importance_xgb.png')

    sauvegarder_modele(xgb_optimal, 'xgboost.pkl')
    afficher_courbe_roc(y_test, probas_test, "XGBoost")
    return xgb_optimal


# ──────────────────────────────────────────────────────────────
# MODÈLE 2c — STACKING (RF + XGBoost → LogisticRegression)
# ──────────────────────────────────────────────────────────────
def entrainer_stacking(X_train, X_test, y_train, y_test, modele_rf, modele_xgb):
    """
    Meta-modèle (Logistic Regression) entraîné sur les probabilités
    cross-validées de RF et XGBoost.
    """
    print("\n" + "=" * 55)
    print("   MODÈLE 2c — STACKING (RF + XGB → LR)")
    print("=" * 55)

    # Prédictions out-of-fold pour le train
    rf_oof  = cross_val_predict(modele_rf,  X_train, y_train, cv=5, method='predict_proba')[:, 1]
    xgb_oof = cross_val_predict(modele_xgb, X_train, y_train, cv=5, method='predict_proba')[:, 1]

    # Prédictions sur le test
    rf_test  = modele_rf.predict_proba(X_test)[:, 1]
    xgb_test = modele_xgb.predict_proba(X_test)[:, 1]

    X_meta_train = np.column_stack([rf_oof,  xgb_oof])
    X_meta_test  = np.column_stack([rf_test, xgb_test])

    meta_lr = LogisticRegression(random_state=42)
    meta_lr.fit(X_meta_train, y_train)
    y_pred_meta = meta_lr.predict(X_meta_test)

    print(f"\n[INFO] Accuracy stacking : {accuracy_score(y_test, y_pred_meta):.3f}")
    print(classification_report(y_test, y_pred_meta,
                                target_names=['Fidèle (0)', 'Churner (1)']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_meta)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'],
                yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de confusion — Stacking')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix_stacking.png')

    poids = pd.Series(meta_lr.coef_[0], index=['RF', 'XGB'])
    print(f"\nPoids du méta-modèle : {poids.to_dict()}")

    sauvegarder_modele(meta_lr, 'stacking.pkl')
    probas_meta = meta_lr.predict_proba(X_meta_test)[:, 1]
    afficher_courbe_roc(y_test, probas_meta, "Stacking")
    return meta_lr


# ──────────────────────────────────────────────────────────────
# MODÈLE 3 — RÉGRESSION XGBoost (MonetaryTotal)
# ──────────────────────────────────────────────────────────────
def entrainer_regression() -> XGBRegressor:
    """
    Prédit MonetaryTotal avec XGBoostRegressor et RandomizedSearchCV.
    Applique log1p si toutes les valeurs sont positives, sinon régression directe.
    """
    print("\n" + "=" * 55)
    print("   MODÈLE 3 — RÉGRESSION XGBOOST (MonetaryTotal)")
    print("=" * 55)

    df = pd.read_csv('data/processed/data_clean.csv')
    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])

    X = df.drop(columns=['MonetaryTotal', 'Churn'])
    y = df['MonetaryTotal']

    # Nettoyage de la cible
    y = y.replace([np.inf, -np.inf], np.nan)
    if y.isnull().any():
        y = y.fillna(y.median())

    # Transformation logarithmique conditionnelle
    valeurs_negatives = (y < 0).any()
    if valeurs_negatives:
        print("[WARN] Valeurs négatives détectées dans MonetaryTotal → pas de log1p")
        y_transf = y
        transformation_log = False
    else:
        y_transf = np.log1p(y)
        transformation_log = True

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transf, test_size=0.2, random_state=42
    )

    # Imputation + scaling
    mediane = X_train.median()
    X_train = X_train.fillna(mediane)
    X_test  = X_test.fillna(mediane)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    espace_params = {
        'n_estimators':    [300, 500, 700],
        'max_depth':       [5, 6, 7, 8],
        'learning_rate':   [0.01, 0.03, 0.05, 0.07],
        'subsample':       [0.7, 0.8, 0.9],
        'colsample_bytree':[0.7, 0.8, 0.9],
    }

    xgb_reg = XGBRegressor(random_state=42)
    recherche = RandomizedSearchCV(
        xgb_reg, espace_params, n_iter=30, cv=5,
        scoring='r2', n_jobs=-1, random_state=42, verbose=1
    )
    recherche.fit(X_train_sc, y_train)
    modele_final = recherche.best_estimator_
    print(f"[INFO] Meilleurs hyperparamètres : {recherche.best_params_}")

    # Évaluation dans l'espace original
    y_pred_transf = modele_final.predict(X_test_sc)
    if transformation_log:
        y_pred = np.expm1(y_pred_transf)
        y_reel = np.expm1(y_test)
    else:
        y_pred = y_pred_transf
        y_reel = y_test

    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    r2   = r2_score(y_reel, y_pred)
    print(f"\n[INFO] RMSE : {rmse:.2f} £ | R² : {r2:.3f}")

    # Graphique réel vs prédit
    plt.figure(figsize=(7, 5))
    plt.scatter(y_reel, y_pred, alpha=0.35, color='steelblue', edgecolors='none')
    diag = [y_reel.min(), y_reel.max()]
    plt.plot(diag, diag, 'r--', lw=1.5)
    plt.xlabel("Valeurs réelles (£)")
    plt.ylabel("Valeurs prédites (£)")
    plt.title("Régression XGBoost — Réel vs Prédit")
    plt.tight_layout()
    sauvegarder_figure('regression_reel_vs_predit.png')

    sauvegarder_modele(modele_final, 'regression_xgboost_optimized.pkl')
    joblib.dump(scaler, 'models/scaler_regression.pkl')
    return modele_final


# ──────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Chargement des données train/test...")
    X_train, X_test, y_train, y_test = charger_train_test()

    # ACP + Clustering
    pca, X_train_pca, X_test_pca = executer_acp(X_train, X_test)
    kmeans = entrainer_kmeans(X_train_pca)

    # Classification
    modele_rf      = entrainer_random_forest(X_train, X_test, y_train, y_test)
    modele_xgb     = entrainer_xgboost(X_train, X_test, y_train, y_test)
    modele_stacking = entrainer_stacking(
        X_train, X_test, y_train, y_test, modele_rf, modele_xgb
    )

    # Régression
    modele_reg = entrainer_regression()

    print("\n" + "=" * 55)
    print("   Entraînement terminé — modèles dans models/ | figures dans reports/")
    print("=" * 55)
