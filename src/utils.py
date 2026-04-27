"""
utils.py 
Fonctions utilitaires : chargement, sauvegarde, visualisation.
Style fonctionnel avec nommage et logs distincts de l'original.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ──────────────────────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────────────────────
def lire_dataset(chemin: str) -> pd.DataFrame:
    """Charge un fichier CSV et affiche ses dimensions."""
    df = pd.read_csv(chemin)
    print(f"[INFO] Dataset chargé — {df.shape[0]} lignes × {df.shape[1]} colonnes ({chemin})")
    return df


def lire_train_test(
    dossier: str = 'data/train_test',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Lit les quatre fichiers CSV du split train/test depuis `dossier`.
    Retourne (X_train, X_test, y_train, y_test).
    """
    X_train = pd.read_csv(f'{dossier}/X_train.csv')
    X_test  = pd.read_csv(f'{dossier}/X_test.csv')
    y_train = pd.read_csv(f'{dossier}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{dossier}/y_test.csv').squeeze()
    print(f"[INFO] Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# Alias pour la compatibilité avec train_model.py / predict.py
charger_train_test = lire_train_test


def recuperer_modele(nom_fichier: str, dossier: str = 'models'):
    """Charge un artefact joblib depuis `dossier`."""
    chemin = f'{dossier}/{nom_fichier}'
    modele = joblib.load(chemin)
    print(f"[INFO] Modèle chargé : {chemin}")
    return modele


# ──────────────────────────────────────────────────────────────
# SAUVEGARDE
# ──────────────────────────────────────────────────────────────
def sauvegarder_modele(modele, nom_fichier: str, dossier: str = 'models') -> None:
    """Sérialise un modèle avec joblib dans `dossier`."""
    os.makedirs(dossier, exist_ok=True)
    chemin = f'{dossier}/{nom_fichier}'
    joblib.dump(modele, chemin)
    print(f"[INFO] Modèle sauvegardé : {chemin}")


def sauvegarder_figure(nom_fichier: str, dossier: str = 'reports', dpi: int = 150) -> None:
    """Enregistre la figure matplotlib courante puis la ferme."""
    os.makedirs(dossier, exist_ok=True)
    chemin = f'{dossier}/{nom_fichier}'
    plt.savefig(chemin, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"[INFO] Figure sauvegardée : {chemin}")


# ──────────────────────────────────────────────────────────────
# VISUALISATIONS
# ──────────────────────────────────────────────────────────────
def afficher_importance_features(
    modele,
    noms_features: list[str],
    top_n: int = 20,
    couleur: str = 'steelblue',
) -> None:
    """
    Trace un bar chart horizontal des `top_n` features les plus importantes.
    Compatible avec tout modèle exposant `feature_importances_`.
    """
    importances = (
        pd.Series(modele.feature_importances_, index=noms_features)
        .sort_values(ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color=couleur, edgecolor='white')
    plt.title(f'Top {top_n} features les plus importantes')
    plt.ylabel('Importance')
    plt.tight_layout()
    sauvegarder_figure('feature_importance.png')


def afficher_distribution_cible(y: pd.Series, titre: str = 'Distribution de Churn') -> None:
    """Camembert de la répartition des classes (0 = Fidèle, 1 = Churner)."""
    counts = y.value_counts()
    labels = [f'Fidèle (0)\n{counts.get(0, 0)}', f'Churner (1)\n{counts.get(1, 0)}']
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%',
            colors=['#4C72B0', '#DD8452'], startangle=90)
    plt.title(titre)
    plt.tight_layout()
    sauvegarder_figure('distribution_churn.png')


def afficher_heatmap_correlation(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Heatmap de corrélation sur les `top_n` colonnes numériques
    les plus corrélées à 'Churn' (si présente, sinon toutes).
    """
    numeriques = df.select_dtypes(include=[np.number])
    if 'Churn' in numeriques.columns:
        corr_churn = numeriques.corr()['Churn'].abs().sort_values(ascending=False)
        cols = corr_churn.head(top_n).index.tolist()
    else:
        cols = numeriques.columns[:top_n].tolist()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        numeriques[cols].corr(),
        annot=False, cmap='coolwarm', center=0,
        linewidths=0.3, square=True,
    )
    plt.title(f'Matrice de corrélation (top {top_n} features)')
    plt.tight_layout()
    sauvegarder_figure('heatmap_correlation.png')
