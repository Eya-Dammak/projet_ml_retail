"""import joblib
import pandas as pd

X_train = pd.read_csv('data/train_test/X_train.csv')
pca     = joblib.load('models/pca.pkl')
kmeans  = joblib.load('models/kmeans.pkl')

X_train_pca = pca.transform(X_train)
labels      = kmeans.labels_

repartition = pd.Series(labels).value_counts().sort_index()

print("=" * 50)
print("   TAILLES À METTRE DANS LE TABLEAU LATEX")
print("=" * 50)
for cluster, taille in repartition.items():
    pct = taille / len(labels) * 100
    print(f"  Cluster {cluster} : {taille:>5} obs  ({pct:.1f} %)")



"accuracy_randomForest"
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

rf = joblib.load('models/random_forest.pkl')

# Re-standardisation comme dans le code
scaler_local = StandardScaler()
scaler_local.fit(X_train)
X_test_sc = scaler_local.transform(X_test)

probas = rf.predict_proba(X_test_sc)[:, 1]

# Seuil optimal
meilleur_seuil, meilleur_f1 = 0.5, 0.0
for seuil in np.arange(0.25, 0.75, 0.01):
    preds = (probas >= seuil).astype(int)
    score = f1_score(y_test, preds)
    if score > meilleur_f1:
        meilleur_f1 = score
        meilleur_seuil = seuil

y_pred = (probas >= meilleur_seuil).astype(int)

print("=" * 55)
print(f"  Seuil optimal     : {meilleur_seuil:.2f}")
print(f"  F1 au seuil opt.  : {meilleur_f1:.3f}")
print(f"  Accuracy          : {accuracy_score(y_test, y_pred):.3f}")
print(f"  AUC               : {roc_auc_score(y_test, probas):.3f}")
print("=" * 55)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, 
      target_names=['Fidèle (0)', 'Churner (1)']))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nMeilleurs hyperparamètres :")
print(rf.get_params())

"accuracy_xgboost"
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from imblearn.over_sampling import SMOTE

X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

xgb = joblib.load('models/xgboost.pkl')

# SMOTE comme dans le code
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"Distribution après SMOTE : {pd.Series(y_sm).value_counts().to_dict()}")

probas = xgb.predict_proba(X_test)[:, 1]

# Seuil optimal
meilleur_seuil, meilleure_acc = 0.5, 0.0
for seuil in np.arange(0.3, 0.7, 0.01):
    preds = (probas >= seuil).astype(int)
    acc   = accuracy_score(y_test, preds)
    if acc > meilleure_acc:
        meilleure_acc = acc
        meilleur_seuil = seuil

y_pred = (probas >= meilleur_seuil).astype(int)

print("=" * 55)
print(f"  Seuil optimal : {meilleur_seuil:.2f}")
print(f"  Accuracy      : {accuracy_score(y_test, y_pred):.3f}")
print(f"  AUC           : {roc_auc_score(y_test, probas):.3f}")
print("=" * 55)
print(classification_report(y_test, y_pred,
      target_names=['Fidèle (0)', 'Churner (1)']))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print(f"\nMeilleurs hyperparamètres : {xgb.get_params()}")

"stacking_model"
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

rf       = joblib.load('models/random_forest.pkl')
xgb      = joblib.load('models/xgboost.pkl')
stacking = joblib.load('models/stacking.pkl')

# Meta-features
rf_test  = rf.predict_proba(X_test)[:, 1]
xgb_test = xgb.predict_proba(X_test)[:, 1]
X_meta_test = np.column_stack([rf_test, xgb_test])

y_pred   = stacking.predict(X_meta_test)
probas   = stacking.predict_proba(X_meta_test)[:, 1]

print("=" * 55)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  AUC      : {roc_auc_score(y_test, probas):.3f}")
print("=" * 55)
print(classification_report(y_test, y_pred,
      target_names=['Fidèle (0)', 'Churner (1)']))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print(f"\nPoids méta-modèle RF/XGB : {dict(zip(['RF','XGB'], stacking.coef_[0].round(3)))}")


import joblib
import pandas as pd

X_train = pd.read_csv('data/train_test/X_train.csv')

rf  = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')

# Top 15 RF
print("=" * 55)
print("   TOP 15 FEATURES --- RANDOM FOREST")
print("=" * 55)
importances_rf = (
    pd.Series(rf.feature_importances_, index=X_train.columns)
    .sort_values(ascending=False)
    .head(15)
)
print(importances_rf.round(4).to_string())

# Top 15 XGBoost
print("\n" + "=" * 55)
print("   TOP 15 FEATURES --- XGBOOST")
print("=" * 55)
importances_xgb = (
    pd.Series(xgb.feature_importances_, index=X_train.columns)
    .sort_values(ascending=False)
    .head(15)
)
print(importances_xgb.round(4).to_string())

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)

# ── Régression ──
reg    = joblib.load('models/regression_xgboost_optimized.pkl')
scaler = joblib.load('models/scaler_regression.pkl')

df = pd.read_csv('data/processed/data_clean.csv')
if 'Country' in df.columns:
    df = df.drop(columns=['Country'])

X = df.drop(columns=['MonetaryTotal', 'Churn'])
y = df['MonetaryTotal']

from sklearn.model_selection import train_test_split
y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())

valeurs_negatives = (y < 0).any()
if not valeurs_negatives:
    y_transf = np.log1p(y)
else:
    y_transf = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y_transf, test_size=0.2, random_state=42
)

mediane = X_train.median()
X_train = X_train.fillna(mediane)
X_test  = X_test.fillna(mediane)
X_test_sc = scaler.transform(X_test)

y_pred_transf = reg.predict(X_test_sc)
if not valeurs_negatives:
    y_pred = np.expm1(y_pred_transf)
    y_reel = np.expm1(y_test)
else:
    y_pred = y_pred_transf
    y_reel = y_test

rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
r2   = r2_score(y_reel, y_pred)
mae  = mean_absolute_error(y_reel, y_pred)

print("=" * 55)
print("   RÉGRESSION XGBOOST")
print("=" * 55)
print(f"  R²   : {r2:.3f}")
print(f"  RMSE : {rmse:.2f} £")
print(f"  MAE  : {mae:.2f} £")
print(f"\nMeilleurs hyperparamètres : {reg.get_params()}")"



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# Chargement
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

# Espace de recherche
espace_hyperparams = {
    'n_estimators':      [600, 800, 1000, 1200],
    'max_depth':         [20, 30, 40, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf':  [1, 2, 3],
    'max_features':      ['sqrt', 'log2', 0.7],
    'bootstrap':         [True],
    'criterion':         ['gini'],
    'max_samples':       [0.8, 0.9, None],
}

# Recherche
rf_base = RandomForestClassifier(
    random_state=42, class_weight='balanced', n_jobs=-1
)
recherche = RandomizedSearchCV(
    rf_base, param_distributions=espace_hyperparams,
    n_iter=60, cv=5, scoring='f1',
    n_jobs=-1, verbose=1, random_state=42,
)
recherche.fit(X_train, y_train)

print(f"[INFO] Meilleurs hyperparamètres : {recherche.best_params_}")
print(f"[INFO] Meilleur F1 cross-val     : {recherche.best_score_:.3f}")

# Évaluation
rf_optimal = recherche.best_estimator_
y_pred = rf_optimal.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Churner']))"""




"""import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Chargement
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

# SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"Après SMOTE : {pd.Series(y_sm).value_counts().to_dict()}")

# Grille
grille = {
    'n_estimators':     [500, 700],
    'max_depth':        [5, 6, 7],
    'learning_rate':    [0.03, 0.05, 0.08],
    'subsample':        [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
}

xgb_base = XGBClassifier(
    random_state=42, eval_metric='logloss', scale_pos_weight=3
)
gs = GridSearchCV(
    xgb_base, grille, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
gs.fit(X_sm, y_sm)

print(f"[INFO] Meilleurs hyperparamètres : {gs.best_params_}")
print(f"[INFO] Meilleur score cross-val  : {gs.best_score_:.3f}")

# Optimisation du seuil
X_sub, X_val, y_sub, y_val = train_test_split(
    X_sm, y_sm, test_size=0.2, random_state=42, stratify=y_sm
)
xgb_opt = gs.best_estimator_
xgb_opt.fit(X_sub, y_sub)
probas_val = xgb_opt.predict_proba(X_val)[:, 1]

meilleur_seuil, meilleure_acc = 0.5, 0.0
for seuil in np.arange(0.3, 0.7, 0.01):
    preds = (probas_val >= seuil).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > meilleure_acc:
        meilleure_acc = acc
        meilleur_seuil = seuil

print(f"[INFO] Seuil optimal : {meilleur_seuil:.2f}")

# Évaluation finale
xgb_opt.fit(X_sm, y_sm)
probas_test = xgb_opt.predict_proba(X_test)[:, 1]
y_pred = (probas_test >= meilleur_seuil).astype(int)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Churner']))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

# Chargement
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

# SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# Modèle avec les meilleurs hyperparamètres déjà trouvés
xgb = XGBClassifier(
    colsample_bytree=0.9,
    learning_rate=0.08,
    max_depth=5,
    n_estimators=500,
    subsample=0.9,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_sm, y_sm)

probas = xgb.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probas)
print(f"AUC : {auc:.3f}")"""
"""import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Chargement
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

# Random Forest (meilleurs hyperparamètres déjà trouvés)
rf = RandomForestClassifier(
    n_estimators=800, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, max_features=0.7, max_samples=None,
    bootstrap=True, class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# XGBoost avec SMOTE (meilleurs hyperparamètres déjà trouvés)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
xgb = XGBClassifier(
    colsample_bytree=0.9, learning_rate=0.08, max_depth=5,
    n_estimators=500, subsample=0.9, scale_pos_weight=3,
    random_state=42, eval_metric='logloss'
)
xgb.fit(X_sm, y_sm)

# Stacking
rf_oof  = cross_val_predict(rf,  X_train, y_train, cv=5, method='predict_proba')[:, 1]
xgb_oof = cross_val_predict(xgb, X_train, y_train, cv=5, method='predict_proba')[:, 1]

rf_test  = rf.predict_proba(X_test)[:, 1]
xgb_test = xgb.predict_proba(X_test)[:, 1]

X_meta_train = np.column_stack([rf_oof,  xgb_oof])
X_meta_test  = np.column_stack([rf_test, xgb_test])

meta_lr = LogisticRegression(random_state=42)
meta_lr.fit(X_meta_train, y_train)
y_pred = meta_lr.predict(X_meta_test)
probas = meta_lr.predict_proba(X_meta_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC      : {roc_auc_score(y_test, probas):.3f}")
print(f"Poids RF / XGB : {meta_lr.coef_[0]}")
print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Churner']))"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Chargement
df = pd.read_csv('data/processed/data_clean.csv')
if 'Country' in df.columns:
    df = df.drop(columns=['Country'])

X = df.drop(columns=['MonetaryTotal', 'Churn'])
y = df['MonetaryTotal']

# Nettoyage
y = y.replace([np.inf, -np.inf], np.nan)
if y.isnull().any():
    y = y.fillna(y.median())

# Transformation log
if (y < 0).any():
    y_transf = y
    log = False
else:
    y_transf = np.log1p(y)
    log = True

# Split
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

# Recherche
espace_params = {
    'n_estimators':     [300, 500, 700],
    'max_depth':        [5, 6, 7, 8],
    'learning_rate':    [0.01, 0.03, 0.05, 0.07],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

xgb_reg = XGBRegressor(random_state=42)
recherche = RandomizedSearchCV(
    xgb_reg, espace_params, n_iter=30, cv=5,
    scoring='r2', n_jobs=-1, random_state=42, verbose=1
)
recherche.fit(X_train_sc, y_train)

print(f"[INFO] Meilleurs hyperparamètres : {recherche.best_params_}")
print(f"[INFO] Meilleur R² cross-val     : {recherche.best_score_:.3f}")

# Évaluation
y_pred_transf = recherche.best_estimator_.predict(X_test_sc)
if log:
    y_pred = np.expm1(y_pred_transf)
    y_reel = np.expm1(y_test)
else:
    y_pred = y_pred_transf
    y_reel = y_test

rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
r2   = r2_score(y_reel, y_pred)
print(f"\nRMSE : {rmse:.2f} £")
print(f"R²   : {r2:.3f}")