import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ============================================================
# 1. Chargement des données
# ============================================================
def charger_donnees(chemin: str) -> pd.DataFrame:
    """Lit le fichier CSV brut et retourne un DataFrame."""
    df = pd.read_csv(chemin)
    print(f"[INFO] Dataset chargé — {df.shape[0]} observations, {df.shape[1]} variables")
    return df

# ============================================================
# 2. Suppression des colonnes non informatives
# ============================================================
def supprimer_cols_non_informatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    - CustomerID  : simple identifiant, sans valeur prédictive
    - NewsletterSubscribed : variance nulle (toujours 'Yes')
    """
    a_supprimer = ['CustomerID', 'NewsletterSubscribed']
    df = df.drop(columns=a_supprimer, errors='ignore')
    print(f"[INFO] Colonnes non informatives supprimées : {a_supprimer}")
    return df

# ============================================================
# 3. Détection et correction des valeurs aberrantes
# ============================================================
def traiter_aberrantes(df: pd.DataFrame) -> pd.DataFrame:
    """
    SupportTicketsCount : 999 et -1 sont des sentinelles hors-plage → NaN
    SatisfactionScore   : 99 et -1 idem
    """
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace({999: np.nan, -1: np.nan})
    df['SatisfactionScore']   = df['SatisfactionScore'].replace({99: np.nan, -1: np.nan})
    print("[INFO] Valeurs aberrantes (SupportTicketsCount, SatisfactionScore) remplacées par NaN")
    return df

# ============================================================
# 4. Parsing de RegistrationDate
# ============================================================
def extraire_features_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la date texte en datetime puis décompose en 4 features numériques :
    RegYear, RegMonth, RegDay, RegWeekday (0=Lundi … 6=Dimanche).
    La colonne originale est supprimée.
    """
    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'], dayfirst=True, errors='coerce'
    )
    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday
    df = df.drop(columns=['RegistrationDate'])
    print("[INFO] RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday")
    return df

# ============================================================
# 5. Transformation de LastLoginIP → IsPrivateIP
# ============================================================
def extraire_feature_ip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte si l'adresse IP est privée (réseau local) ou publique.
    Retourne 1 si privée, 0 sinon. La colonne IP brute est supprimée.
    """
    def _est_prive(ip_str: str) -> int:
        try:
            return int(ipaddress.ip_address(str(ip_str)).is_private)
        except ValueError:
            return 0

    df['IsPrivateIP'] = df['LastLoginIP'].apply(_est_prive)
    df = df.drop(columns=['LastLoginIP'])
    print("[INFO] LastLoginIP → IsPrivateIP (1=privé, 0=public)")
    return df

# ============================================================
# 6. Feature engineering
# ============================================================
def creer_nouvelles_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trois nouvelles variables construites AVANT la suppression anti-leakage :
      - MonetaryPerDay  : dépense moyenne par jour d'inactivité
      - AvgBasketValue  : montant moyen par commande
      - TenureRatio     : ratio inactivité / ancienneté (comportement récent vs historique)
    """
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    df['TenureRatio']    = df['Recency'] / (df['CustomerTenureDays'] + 1)
    print("[INFO] Nouvelles features créées : MonetaryPerDay, AvgBasketValue, TenureRatio")
    return df

# ============================================================
# 7. Suppression des features à risque de data leakage
# ============================================================
def supprimer_features_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes dont l'information est :
      - dérivée de Churn (ChurnRiskCategory, AccountStatus…)
      - résumée dans les nouvelles features (CustomerTenureDays, Recency, TenureRatio)
      - trop corrélées à la cible pour un usage réaliste
    """
    leakage = [
        'ChurnRiskCategory', 'CustomerType', 'LoyaltyLevel',
        'SpendingCategory', 'RFMSegment', 'AccountStatus',
        'ReturnRatio', 'NegQtyCount', 'ZeroPriceCount',
        'CancelledTransactions', 'CustomerTenureDays',
        'FirstPurchase', 'Age', 'SupportTicketsCount',
        'SatisfactionScore', 'Recency', 'TenureRatio',
    ]
    df = df.drop(columns=leakage, errors='ignore')
    print(f"[INFO] Features leakage supprimées ({len(leakage)} colonnes)")
    return df

# ============================================================
# 8. Réduction de la multicolinéarité (seuil |r| > 0.8)
# ============================================================
def reduire_multicolinearite(df: pd.DataFrame, seuil: float = 0.8) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation absolue sur les features numériques (hors Churn).
    Pour chaque paire dépassant le seuil, la première colonne est retirée.
    """
    numeriques = [c for c in df.select_dtypes(include=['float64', 'int64']).columns
                  if c != 'Churn']
    if not numeriques:
        print("[WARN] Aucune colonne numérique détectée pour l'analyse de multicolinéarité")
        return df

    matrice_corr = df[numeriques].corr().abs()
    a_retirer = set()
    for i in range(len(matrice_corr.columns)):
        for j in range(i):
            if matrice_corr.iloc[i, j] > seuil:
                col_redondante = matrice_corr.columns[i]
                a_retirer.add(col_redondante)
                print(f"   [CORR] {col_redondante} ↔ {matrice_corr.columns[j]} "
                      f"= {matrice_corr.iloc[i, j]:.2f} → retrait de {col_redondante}")

    df = df.drop(columns=list(a_retirer))
    print(f"[INFO] Multicolinéarité : {len(a_retirer)} colonne(s) retirée(s)")
    return df

# ============================================================
# 9. Encodage des variables catégorielles (hors Country)
# ============================================================
def encoder_variables_categorielles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodage ordinal pour les variables avec ordre naturel,
    One-Hot pour les variables nominales sans ordre (sauf Country, traitée après split).
    """
    # --- Encodage ordinal ---
    configs_ordinales = {
        'AgeCategory': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }
    for col, ordre in configs_ordinales.items():
        if col in df.columns:
            enc = OrdinalEncoder(
                categories=[ordre],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            df[col] = enc.fit_transform(df[[col]])
            print(f"   [ENC] Ordinal : {col}")

    # --- Encodage One-Hot (nominales, hors Country) ---
    cols_onehot = [c for c in ['FavoriteSeason', 'Region', 'WeekendPreference',
                                'ProductDiversity', 'Gender'] if c in df.columns]
    df = pd.get_dummies(df, columns=cols_onehot, drop_first=False)
    print(f"[INFO] One-Hot : {cols_onehot}")
    return df

# ============================================================
# 10. Split, One-Hot Country, Imputation, Standardisation
# ============================================================
def preparer_jeux_train_test(df: pd.DataFrame):
    """
    Pipeline final :
      1. Séparation stratifiée 80/20
      2. One-Hot encoding de Country (fit sur union train+test pour couvrir tous les pays)
      3. Imputation par la médiane (calculée sur X_train uniquement)
      4. Standardisation (StandardScaler fitté sur X_train)
      5. Sauvegarde des CSV et artefacts (.pkl)
    """
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Séparation → train : {X_train.shape} | test : {X_test.shape}")

    # --- One-Hot de Country (après split pour éviter la fuite) ---
    if 'Country' in X_train.columns:
        combined_country = pd.concat(
            [X_train[['Country']], X_test[['Country']]], axis=0
        )
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(combined_country[['Country']])

        country_cols = [f'Country_{c}' for c in ohe.categories_[0]]
        train_ohe = pd.DataFrame(ohe.transform(X_train[['Country']]),
                                 columns=country_cols)
        test_ohe  = pd.DataFrame(ohe.transform(X_test[['Country']]),
                                 columns=country_cols)

        X_train = pd.concat(
            [X_train.drop(columns=['Country']).reset_index(drop=True), train_ohe], axis=1
        )
        X_test = pd.concat(
            [X_test.drop(columns=['Country']).reset_index(drop=True), test_ohe], axis=1
        )
        print(f"[INFO] One-Hot Country : {len(ohe.categories_[0])} modalités")
    else:
        print("[WARN] Colonne 'Country' absente du dataset")

    # --- Imputation par la médiane du train ---
    mediane = X_train.median()
    X_train = X_train.fillna(mediane)
    X_test  = X_test.fillna(mediane)
    print("[INFO] Imputation médiane (paramètres issus du train uniquement)")

    # --- Standardisation ---
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    print("[INFO] StandardScaler appliqué (fit sur train)")

    # --- Sauvegarde ---
    os.makedirs('data/train_test', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    X_train_sc.to_csv('data/train_test/X_train.csv', index=False)
    X_test_sc.to_csv('data/train_test/X_test.csv',   index=False)
    y_train.to_csv('data/train_test/y_train.csv',     index=False)
    y_test.to_csv('data/train_test/y_test.csv',       index=False)
    joblib.dump(scaler,  'models/scaler.pkl')
    joblib.dump(mediane, 'models/mediane_train.pkl')
    print("[INFO] Artefacts sauvegardés dans data/train_test/ et models/")

    return X_train_sc, X_test_sc, y_train, y_test

# ============================================================
# Pipeline principal
# ============================================================
if __name__ == "__main__":
    # Étape 1 – Chargement
    df = charger_donnees('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')

    # Étapes 2-9 – Nettoyage & transformation
    df = supprimer_cols_non_informatives(df)
    df = traiter_aberrantes(df)
    df = extraire_features_date(df)
    df = extraire_feature_ip(df)
    df = creer_nouvelles_features(df)
    df = supprimer_features_leakage(df)
    df = reduire_multicolinearite(df)
    df = encoder_variables_categorielles(df)

    # Sauvegarde intermédiaire
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/data_clean.csv', index=False)
    print(f"\n[INFO] data_clean.csv enregistré — dimensions finales : {df.shape}")

    # Étape 10 – Split, imputation, scaling
    X_train, X_test, y_train, y_test = preparer_jeux_train_test(df)

    print("\n[DONE] Preprocessing terminé avec succès !")
