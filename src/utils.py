"""
utils.py — Fonctions utilitaires réutilisables
Projet : Analyse Comportementale Clientèle Retail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1. Résumé global du dataset
# ============================================================

def resume_dataset(df):
    """Affiche un résumé complet du DataFrame."""
    print("=" * 55)
    print("  RÉSUMÉ DATASET")
    print("=" * 55)
    print(f"  Lignes      : {df.shape[0]:,}")
    print(f"  Colonnes    : {df.shape[1]}")
    print(f"  Doublons    : {df.duplicated().sum()}")
    print(f"  Mémoire     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Total NaN   : {df.isnull().sum().sum():,}")
    print("\n  Types de colonnes :")
    print(df.dtypes.value_counts().to_string())
    print("=" * 55)


# ============================================================
# 2. Valeurs manquantes
# ============================================================

def afficher_valeurs_manquantes(df):
    """
    Analyse et affiche les valeurs manquantes par colonne.
    Retourne un DataFrame trié par pourcentage décroissant.
    """
    if df.empty:
        print("⚠️  DataFrame vide.")
        return None

    manquantes  = df.isnull().sum()
    pourcentage = (manquantes / len(df)) * 100

    resume = pd.DataFrame({
        "Valeurs_manquantes": manquantes,
        "Pourcentage (%)":    pourcentage.round(2)
    })
    resume = resume[resume["Valeurs_manquantes"] > 0]
    resume = resume.sort_values("Pourcentage (%)", ascending=False)

    if resume.empty:
        print("✅  Aucune valeur manquante détectée.")
    else:
        print(f"⚠️  {len(resume)} colonnes contiennent des NaN :\n")
        print(resume.to_string())
    return resume


# ============================================================
# 3. Détection des Outliers (IQR)
# ============================================================

def detecter_outliers_iqr(df, colonne):
    """
    Détecte les outliers d'une colonne numérique via la méthode IQR.
    Retourne : (n_outliers, borne_basse, borne_haute)
    """
    if colonne not in df.columns:
        raise ValueError(f"Colonne '{colonne}' introuvable dans le DataFrame.")
    if not np.issubdtype(df[colonne].dtype, np.number):
        raise TypeError(f"Colonne '{colonne}' n'est pas numérique.")

    Q1  = df[colonne].quantile(0.25)
    Q3  = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    outliers = df[(df[colonne] < borne_basse) | (df[colonne] > borne_haute)]

    print("=" * 45)
    print(f"  Analyse outliers : {colonne}")
    print(f"  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
    print(f"  Borne basse : {borne_basse:.2f}")
    print(f"  Borne haute : {borne_haute:.2f}")
    print(f"  Outliers    : {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print("=" * 45)
    return len(outliers), borne_basse, borne_haute


def detecter_outliers_toutes_colonnes(df):
    """
    Applique la détection IQR sur toutes les colonnes numériques.
    Retourne un DataFrame résumé trié par nombre d'outliers.
    """
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    resultats = []
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb  = Q1 - 1.5 * IQR
        ub  = Q3 + 1.5 * IQR
        n   = ((df[col] < lb) | (df[col] > ub)).sum()
        resultats.append({
            "feature":      col,
            "borne_basse":  round(lb, 2),
            "borne_haute":  round(ub, 2),
            "n_outliers":   n,
            "pct_outliers": round(n / len(df) * 100, 2)
        })
    return pd.DataFrame(resultats).sort_values("n_outliers", ascending=False)


# ============================================================
# 4. Visualisations
# ============================================================

def afficher_distribution(df, colonne, bins=30, output_dir=None):
    """Histogramme + Boxplot pour une colonne numérique."""
    if colonne not in df.columns:
        raise ValueError(f"Colonne '{colonne}' introuvable.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[colonne].dropna(), bins=bins, kde=True,
                 ax=axes[0], color="#1e88e5")
    axes[0].set_title(f"Distribution — {colonne}")

    sns.boxplot(x=df[colonne].dropna(), ax=axes[1], color="#43a047")
    axes[1].set_title(f"Boxplot — {colonne}")

    plt.tight_layout()
    if output_dir:
        sauvegarder_graphique(f"dist_{colonne}.png", output_dir)
    plt.show()


def afficher_distributions_multiples(df, cols, output_dir=None):
    """Grille d'histogrammes pour plusieurs colonnes numériques."""
    n     = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Distributions des features numériques", fontsize=12, y=1.01)
    plt.tight_layout()
    if output_dir:
        sauvegarder_graphique("distributions_all.png", output_dir)
    plt.show()


def afficher_correlation(df, seuil=0.8, heatmap=True, output_dir=None):
    """
    Affiche la heatmap de corrélation et retourne les paires fortement corrélées.
    """
    df_num = df.select_dtypes(include=np.number)
    if df_num.empty:
        print("⚠️  Aucune colonne numérique.")
        return None

    corr = df_num.corr().abs()

    if heatmap:
        plt.figure(figsize=(18, 14))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                    annot=False, linewidths=0.3)
        plt.title("Matrice de corrélation", fontsize=14)
        plt.tight_layout()
        if output_dir:
            sauvegarder_graphique("correlation_heatmap.png", output_dir)
        plt.show()

    paires = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] >= seuil:
                paires.append({
                    "Feature_1":   corr.columns[i],
                    "Feature_2":   corr.columns[j],
                    "Corrélation": round(corr.iloc[i, j], 3)
                })
    if paires:
        result = pd.DataFrame(paires).sort_values("Corrélation", ascending=False)
        print(f"\n⚠️  {len(result)} paires avec corrélation ≥ {seuil} :")
        print(result.to_string(index=False))
        return result
    else:
        print(f"✅  Aucune corrélation ≥ {seuil}")
        return None


# ============================================================
# 5. Sauvegarde graphique
# ============================================================

def sauvegarder_graphique(nom_fichier, dossier="../reports"):
    """Sauvegarde le graphique matplotlib actif dans le dossier reports."""
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, nom_fichier)
    plt.savefig(chemin, bbox_inches="tight", dpi=150)
    print(f"✅  Graphique sauvegardé → {chemin}")


# ============================================================
# 6. Évaluation modèle de classification
# ============================================================

from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)

def evaluer_classificateur(model, X_test, y_test,
                            nom_modele="Modèle", output_dir=None):
    """
    Affiche rapport de classification, matrice de confusion et courbe ROC.
    Retourne les prédictions.
    """
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)

    print(f"\n{'='*55}")
    print(f"  Évaluation : {nom_modele}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Fidèle (0)", "Churné (1)"]))
    if y_prob is not None:
        print(f"  AUC-ROC : {roc_auc_score(y_test, y_prob):.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fidèle", "Churné"],
                yticklabels=["Fidèle", "Churné"])
    plt.title(f"Confusion Matrix — {nom_modele}")
    plt.ylabel("Réel"); plt.xlabel("Prédit")
    plt.tight_layout()
    if output_dir:
        sauvegarder_graphique(f"confusion_{nom_modele.replace(' ','_')}.png", output_dir)
    plt.show()

    # Courbe ROC
    if y_prob is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob, name=nom_modele)
        plt.title(f"Courbe ROC — {nom_modele}")
        if output_dir:
            sauvegarder_graphique(f"roc_{nom_modele.replace(' ','_')}.png", output_dir)
        plt.show()

    return y_pred
