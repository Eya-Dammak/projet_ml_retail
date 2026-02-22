import pandas as pd          #
import numpy as np           
df = pd.read_csv("C:/pojet_ml/projet_ml_retail/data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
#cleaned_dataset

import numpy as np
import os

df_cleaned = df.copy()

print("Shape initiale :", df_cleaned.shape)


#Supprimer colonnes constantes (1 seule valeur)

cols_const = [col for col in df_cleaned.columns if df_cleaned[col].nunique(dropna=False) <= 1]

df_cleaned.drop(columns=cols_const, inplace=True)

print("Colonnes constantes supprimées :", cols_const)


#Supprimer colonnes quasi-constantes (>95% même valeur)

cols_quasi_const = []

for col in df_cleaned.columns:
    freq_max = df_cleaned[col].value_counts(normalize=True, dropna=False).max()
    if freq_max > 0.95:
        cols_quasi_const.append(col)

df_cleaned.drop(columns=cols_quasi_const, inplace=True)

print("Colonnes quasi-constantes supprimées :", cols_quasi_const)


#Supprimer colonnes inutiles métier

cols_inutiles = [
    'CustomerID',
    'LastLoginIP',
    'RegistrationDate',
    'IP_privee',
    'Country_encoded',
    'UniqueCountries',
    'ZeroPriceCount'
]

cols_inutiles = [col for col in cols_inutiles if col in df_cleaned.columns]

df_cleaned.drop(columns=cols_inutiles, inplace=True)

print("Colonnes inutiles supprimées :", cols_inutiles)


#Supprimer colonnes avec trop de NaN (>40%)

seuil_nan = 0.4
cols_nan = df_cleaned.columns[df_cleaned.isnull().mean() > seuil_nan]

df_cleaned.drop(columns=cols_nan, inplace=True)

print("Colonnes avec trop de NaN supprimées :", list(cols_nan))


#Supprimer colonnes très corrélées (>0.9)

# Sélectionner uniquement colonnes numériques
df_numeric = df_cleaned.select_dtypes(include=[np.number])

corr_matrix = df_numeric.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

cols_corr = [column for column in upper.columns if any(upper[column] > 0.9)]

# ⚠ Ne jamais supprimer la target
if 'Churn' in cols_corr:
    cols_corr.remove('Churn')

df_cleaned.drop(columns=cols_corr, inplace=True)

print("Colonnes très corrélées supprimées :", cols_corr)


#Résultat final

print("\nShape finale :", df_cleaned.shape)
print("\nColonnes restantes :")
print(df_cleaned.columns.tolist())

#Sauvegarde
os.makedirs('../data/processed', exist_ok=True)

df_cleaned.to_csv('../data/processed/data_cleaned_final.csv', index=False)

print("\n✅ Dataset nettoyé sauvegardé dans data/processed/data_cleaned_final.csv")