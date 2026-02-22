import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display

warnings.filterwarnings('ignore')

print("✅ Bibliothèques importées avec succès !")
# ============================================================
# CELLULE 2 : Chargement des données
# ============================================================

# On charge le fichier CSV depuis le dossier data/raw/
df = pd.read_csv("C:/pojet_ml/projet_ml_retail/data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
# Afficher les dimensions du tableau
print(f" Le dataset contient : {df.shape[0]} lignes et {df.shape[1]} colonnes")
print(f"Nom des colonnes :\n{list(df.columns)}")
# ============================================================
# CELLULE 3 : Premier aperçu
# ============================================================

print("=== 5 premières lignes ===")
display(df.head())

print("\n=== 5 dernières lignes ===")
display(df.tail())

print("\n=== Aperçu aléatoire de 5 lignes ===")
display(df.sample(5, random_state=42))
print("=== Types de données de chaque colonne ===")
df.info()
# ============================================================
# CELLULE 5 : Statistiques descriptives
# ============================================================

print("=== Statistiques des colonnes NUMÉRIQUES ===")
display(df.describe())


print("\n=== Statistiques des colonnes CATÉGORIELLES ===")
display(df.describe(include='object'))
# ============================================================
# CELLULE 6 : Valeurs manquantes (NaN)
# ============================================================

# Compter les valeurs manquantes par colonne
valeurs_manquantes = df.isnull().sum()
pourcentage = (df.isnull().sum() / len(df)) * 100

# Créer un tableau récapitulatif
resume_nan = pd.DataFrame({
    'Valeurs manquantes': valeurs_manquantes,
    'Pourcentage (%)': pourcentage.round(2)
})

# Afficher seulement les colonnes qui ont des NaN
resume_nan = resume_nan[resume_nan['Valeurs manquantes'] > 0].sort_values('Pourcentage (%)', ascending=False)

print("=== Colonnes avec des valeurs manquantes ===")
display(resume_nan)

# Visualisation graphique
plt.figure(figsize=(10, 5))
plt.bar(resume_nan.index, resume_nan['Pourcentage (%)'], color='blue')
plt.title('Pourcentage de valeurs manquantes par colonne')
plt.xlabel('Colonnes')
plt.ylabel('Pourcentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../reports/valeurs_manquantes.png')  # Sauvegarder le graphique
plt.show()
print("✅ Graphique sauvegardé dans reports/")
nb_doublons = df.duplicated().sum()
print(f"🔍 Nombre de lignes dupliquées : {nb_doublons}")

if nb_doublons > 0:
    print("Voici un aperçu des doublons :")
    display(df[df.duplicated(keep=False)].head(10))
else:
    print("✅ Aucun doublon détecté !")
    # ============================================================
# CELLULE 8 : Valeurs aberrantes (outliers)
# ============================================================

print("=== SupportTicketsCount ===")
print(df['SupportTicketsCount'].value_counts().sort_index())

print("\n=== SatisfactionScore ===")
print(df['SatisfactionScore'].value_counts().sort_index())

# Visualisation boxplot pour voir les outliers
colonnes_numeriques = [
    'MonetaryTotal', 'Recency', 'Frequency', 
    'TotalQuantity', 'SupportTicketsCount', 
    'SatisfactionScore', 'Age'
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(colonnes_numeriques):
    if col in df.columns:
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col)
        axes[i].set_ylabel('Valeurs')

plt.suptitle('Boxplots - Détection des valeurs aberrantes', fontsize=14)
plt.tight_layout()
plt.savefig('../reports/boxplots_outliers.png')
plt.show()

print("✅ Graphique sauvegardé dans reports/")
# ============================================================
# CELLULE 9 : Distribution de la variable cible CHURN
# ============================================================

# Churn = 0 signifie client fidèle, Churn = 1 = client parti
print("=== Distribution du Churn ===")
print(df['Churn'].value_counts())
print(f"\nPourcentage de clients partis : {df['Churn'].mean()*100:.2f}%")

# Graphique
plt.figure(figsize=(6, 4))
df['Churn'].value_counts().plot(kind='bar', color=['blue', 'green'])
plt.title('Distribution du Churn (0=Fidèle, 1=Parti)')
plt.xlabel('Churn')
plt.ylabel('Nombre de clients')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../reports/distribution_churn.png')
plt.show()
# ============================================================
# CELLULE 10 : Corrélation entre les features numériques
# ============================================================

# Sélectionner uniquement les colonnes numériques
df_numerique = df.select_dtypes(include=[np.number])

# Calculer la matrice de corrélation
matrice_corr = df_numerique.corr()

# Afficher sous forme de heatmap (carte de chaleur)
plt.figure(figsize=(18, 14))
sns.heatmap(
    matrice_corr,
    annot=False,       # Ne pas afficher les chiffres (trop de colonnes)
    cmap='coolwarm',   # Rouge = corrélation positive, Bleu = négative
    center=0,
    vmin=-1, vmax=1
)
plt.title('Matrice de corrélation des features numériques', fontsize=14)
plt.tight_layout()
plt.savefig('../reports/matrice_correlation.png')
plt.show()

# Trouver les paires très corrélées (> 0.8 ou < -0.8)
print("=== Paires de features très corrélées (|corr| > 0.8) ===")
corr_haute = []
for i in range(len(matrice_corr.columns)):
    for j in range(i+1, len(matrice_corr.columns)):
        val = matrice_corr.iloc[i, j]
        if abs(val) > 0.8:
            corr_haute.append({
                'Feature 1': matrice_corr.columns[i],
                'Feature 2': matrice_corr.columns[j],
                'Corrélation': round(val, 3)
            })

if corr_haute:
    display(pd.DataFrame(corr_haute).sort_values('Corrélation', ascending=False))
else:
    print("Aucune paire avec corrélation > 0.8")

# ============================================================
# CELLULE 11 : Colonnes catégorielles
# ============================================================

colonnes_cat = df.select_dtypes(include='object').columns.tolist()
print(f"Colonnes catégorielles : {colonnes_cat}")

for col in colonnes_cat:
    print(f"\n=== {col} ===")
    print(df[col].value_counts())