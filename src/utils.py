# ============================================================
# src/utils.py
# Fonctions utilitaires réutilisables dans tout le projet
# Version améliorée
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1️⃣ Valeurs manquantes
# ============================================================

def afficher_valeurs_manquantes(df):
    """
    Analyse et affiche les valeurs manquantes.

    Retourne un dataframe trié par pourcentage décroissant.
    """
    if df.empty:
        print("⚠️ DataFrame vide.")
        return None

    valeurs_manquantes = df.isnull().sum()
    pourcentage = (valeurs_manquantes / len(df)) * 100

    resume = pd.DataFrame({
        "Valeurs_manquantes": valeurs_manquantes,
        "Pourcentage (%)": pourcentage.round(2)
    })

    resume = resume[resume["Valeurs_manquantes"] > 0]
    resume = resume.sort_values("Pourcentage (%)", ascending=False)

    if resume.empty:
        print("✅ Aucune valeur manquante détectée.")
    else:
        print(f"⚠️ {len(resume)} colonnes contiennent des NaN.")

    return resume


# ============================================================
# 2️⃣ Détection Outliers (IQR)
# ============================================================

def detecter_outliers_iqr(df, colonne):
    """
    Détecte les outliers via méthode IQR.
    Fonctionne uniquement pour colonnes numériques.
    """

    if colonne not in df.columns:
        raise ValueError(f"Colonne '{colonne}' inexistante.")

    if not np.issubdtype(df[colonne].dtype, np.number):
        raise TypeError(f"Colonne '{colonne}' non numérique.")

    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1

    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    outliers = df[
        (df[colonne] < borne_basse) |
        (df[colonne] > borne_haute)
    ]

    print("="*40)
    print(f"📊 Analyse : {colonne}")
    print(f"Borne basse : {borne_basse:.2f}")
    print(f"Borne haute : {borne_haute:.2f}")
    print(f"Outliers    : {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print("="*40)

    return len(outliers), borne_basse, borne_haute


# ============================================================
# 3️⃣ Sauvegarde Graphique
# ============================================================

def sauvegarder_graphique(nom_fichier, dossier="../reports"):
    """
    Sauvegarde le graphique actuel.
    """
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, nom_fichier)

    plt.savefig(chemin, bbox_inches="tight", dpi=200)
    print(f"✅ Graphique sauvegardé : {chemin}")


# ============================================================
# 4️⃣ Distribution d'une variable
# ============================================================

def afficher_distribution(df, colonne, bins=30):
    """
    Affiche histogramme + boxplot.
    """

    if colonne not in df.columns:
        raise ValueError("Colonne inexistante.")

    if not np.issubdtype(df[colonne].dtype, np.number):
        raise TypeError("Colonne non numérique.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[colonne], bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution de {colonne}")

    sns.boxplot(x=df[colonne], ax=axes[1])
    axes[1].set_title(f"Boxplot de {colonne}")

    plt.tight_layout()
    plt.show()


# ============================================================
# 5️⃣ Résumé global
# ============================================================

def resume_dataset(df):
    """
    Résumé global du dataset.
    """

    print("="*50)
    print("📊 RÉSUMÉ DATASET")
    print("="*50)

    print(f"Lignes      : {df.shape[0]:,}")
    print(f"Colonnes    : {df.shape[1]}")
    print(f"Doublons    : {df.duplicated().sum()}")
    print(f"Mémoire     : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

    print("\nTypes colonnes :")
    print(df.dtypes.value_counts())

    print(f"\nTotal NaN : {df.isnull().sum().sum():,}")
    print("="*50)


# ============================================================
# 6️⃣ Corrélation
# ============================================================

def afficher_correlation(df, seuil=0.8, heatmap=False):
    """
    Affiche les paires fortement corrélées.
    Option heatmap possible.
    """

    df_num = df.select_dtypes(include=np.number)

    if df_num.empty:
        print("⚠️ Aucune colonne numérique.")
        return None

    corr = df_num.corr().abs()

    if heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Matrice de corrélation")
        plt.show()

    paires = []

    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] >= seuil:
                paires.append({
                    "Feature 1": corr.columns[i],
                    "Feature 2": corr.columns[j],
                    "Corrélation": round(corr.iloc[i, j], 3)
                })

    if paires:
        result = pd.DataFrame(paires).sort_values("Corrélation", ascending=False)
        print(f"⚠️ {len(result)} paires ≥ {seuil}")
        print(result.to_string(index=False))
        return result
    else:
        print(f"✅ Aucune corrélation ≥ {seuil}")
        return None