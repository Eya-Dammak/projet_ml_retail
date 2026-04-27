# Projet ML — Analyse Comportementale Clientèle Retail

> Pipeline Machine Learning complet pour la prédiction du churn et l'estimation de la valeur client dans un contexte e-commerce retail.

---

## Objectif

Ce projet vise à analyser les comportements clients d'un e-commerce retail (4 372 clients, 52 variables) afin de :
- **Segmenter** les clients en profils distincts (K-Means + ACP)
- **Prédire** le risque de churn (Random Forest, XGBoost, Stacking)
- **Estimer** la valeur économique de chaque client (XGBoost Régression)
- **Déployer** les modèles dans une application web Flask

---

## Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/                  # Données brutes originales (CSV)
│   ├── processed/            # Données nettoyées (data_clean.csv)
│   └── train_test/           # Données splittées (X_train, X_test, y_train, y_test)
├── notebooks/
│   └── prototypage.ipynb     # Exploration des données (EDA)
├── src/
│   ├── preprocessing.py      # Pipeline de nettoyage et préparation
│   ├── train_model.py        # Entraînement des modèles
│   ├── predict.py            # Inférence et prédiction
│   └── utils.py              # Fonctions utilitaires
├── models/                   # Modèles sauvegardés (.pkl)
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── stacking.pkl
│   ├── kmeans.pkl
│   ├── pca.pkl
│   ├── regression_xgboost_optimized.pkl
│   ├── scaler.pkl
│   └── scaler_regression.pkl
├── app/                      # Application web Flask
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── dashboard.html
├── reports/                  # Graphiques et visualisations
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Pipeline ML

### 1. Exploration des données (`notebooks/prototypage.ipynb`)
- Aperçu général, statistiques descriptives
- Détection des valeurs manquantes, doublons, outliers
- Analyse des corrélations et multicolinéarité (VIF)
- Rapport de qualité — 7 problèmes détectés

### 2. Prétraitement (`src/preprocessing.py`)
Pipeline en 10 étapes :
1. Chargement des données brutes
2. Suppression des colonnes non informatives (`CustomerID`, `NewsletterSubscribed`)
3. Correction des valeurs sentinelles (999 / 99 / -1 → NaN)
4. Parsing de `RegistrationDate` → 4 features numériques
5. Transformation de `LastLoginIP` → `IsPrivateIP`
6. Feature Engineering (`MonetaryPerDay`, `AvgBasketValue`, `TenureRatio`)
7. Suppression anti-data leakage (17 colonnes)
8. Réduction de la multicolinéarité (seuil |r| > 0.8)
9. Encodage ordinal + One-Hot
10. Split 80/20, imputation médiane, standardisation

### 3. Entraînement (`src/train_model.py`)

| Modèle | Méthode | Résultat |
|--------|---------|----------|
| ACP | PCA — 95% variance | Réduction dimensionnelle |
| K-Means | k=4, silhouette | 4 segments clients |
| Random Forest | RandomizedSearchCV (60 iter, cv=5) | Accuracy 94.4% |
| XGBoost | GridSearchCV (72 iter, cv=5) + SMOTE | Accuracy 97%, AUC 0.994 |
| Stacking (RF + XGB → LR) | Cross-val out-of-fold | Accuracy 96.8%, AUC 0.992 |
| Régression XGBoost | RandomizedSearchCV (30 iter, cv=5) | R² 0.549, RMSE 7621 £ |

---

## Résultats des modèles de classification

| Modèle | Accuracy | AUC | F1 Churner | Précision Churner |
|--------|----------|-----|------------|-------------------|
| Random Forest | 94.4% | — | 91% | 95% |
| **XGBoost** ✅ | **97%** | **0.994** | **95%** | **99%** |
| Stacking | 96.8% | 0.992 | 95% | 97% |

> **Modèle retenu : XGBoost** — meilleur AUC et meilleure précision churner, plus simple à déployer en production.

---

## Installation

```bash
# Cloner le projet
git clone https://github.com/Eya-Dammak/projet_ml_retail.git
cd projet_ml_retail

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Prétraitement
```bash
python src/preprocessing.py
```

### Entraînement
```bash
python src/train_model.py
```

### Prédiction
```bash
python src/predict.py
```

### Lancer l'application Flask
```bash
python app/app.py
```
Puis ouvrir `http://localhost:5000` dans le navigateur.

---

## Application Web

L'application Flask expose 4 routes principales :

| Route | Fonction |
|-------|----------|
| `/` | Interface de prédiction client |
| `/dashboard` | Dashboard analytics |
| `/predict` | Prédiction churn + segment + revenu estimé |
| `/metrics` | Métriques XGBoost en temps réel |

**Inputs (4 variables) :** Fréquence d'achat · Montant total · Saison préférée · Diversité produits

**Outputs :** Probabilité de churn (RF / XGBoost / Stacking) · Segment K-Means · Revenu estimé en £

---

## Dépendances principales

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
flask
joblib
matplotlib
seaborn
statsmodels
```

---

## Auteur

**Eya Damak** — GI2S4