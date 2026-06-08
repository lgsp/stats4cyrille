"""
STA211 - Analyse : Arbres de régression/classification & Méta-algorithmes
Préparé pour le samedi 6 juin 2026
Inspiré des cours de V. Audigier & N. Niang
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.datasets import fetch_openml

# ---- 1. Chargement des données (German Credit) ----
print("=" * 60)
print("CHARGEMENT DES DONNÉES - German Credit")
print("=" * 60)

data = fetch_openml(data_id=31, as_frame=True, parser='pandas')
df = data.frame

# La cible est 'class' (bon/mauvais payeur)
# Renommer pour clarté
df.rename(columns={'class': 'Creditability'}, inplace=True)
df['Creditability'] = (df['Creditability'] == 'good').astype(int)

# Encodage one-hot des variables catégorielles
df_encoded = pd.get_dummies(df.drop(columns=['Creditability']), drop_first=True)
df_encoded['Creditability'] = df['Creditability'].values

X = df_encoded.drop(columns=['Creditability'])
y = df_encoded['Creditability']

# Séparation apprentissage/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=235
)

print(f"Dimensions apprentissage: {X_train.shape}")
print(f"Dimensions test: {X_test.shape}")

# ---- 2. Arbre CART ----
print("\n" + "=" * 60)
print("ARBRE DE DÉCISION CART")
print("=" * 60)

tree = DecisionTreeClassifier(random_state=235, min_samples_leaf=5)
tree.fit(X_train, y_train)

y_score_tree = tree.predict_proba(X_test)[:, 1]
y_pred_tree  = tree.predict(X_test)

auc_tree = roc_auc_score(y_test, y_score_tree)
err_tree = 1 - accuracy_score(y_test, y_pred_tree)

print(f"AUC CART:             {auc_tree:.4f}")
print(f"Erreur CART:          {err_tree:.4f}")

# ---- 3. Bagging ----
print("\n" + "=" * 60)
print("BAGGING")
print("=" * 60)

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(min_samples_leaf=5, random_state=235),
    n_estimators=200,
    random_state=235,
    oob_score=True,
    bootstrap=True
)
bag.fit(X_train, y_train)

y_score_bag = bag.predict_proba(X_test)[:, 1]
auc_bag = roc_auc_score(y_test, y_score_bag)
err_bag = 1 - accuracy_score(y_test, bag.predict(X_test))

print(f"AUC Bagging:          {auc_bag:.4f}")
print(f"Erreur Bagging:       {err_bag:.4f}")
print(f"Erreur OOB:           {1 - bag.oob_score_:.4f}")

# ---- 4. Forêt aléatoire ----
print("\n" + "=" * 60)
print("FORÊT ALÉATOIRE")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    random_state=235,
    oob_score=True
)
rf.fit(X_train, y_train)

y_score_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_score_rf)
err_rf = 1 - accuracy_score(y_test, rf.predict(X_test))

print(f"AUC Random Forest:    {auc_rf:.4f}")
print(f"Erreur Random Forest: {err_rf:.4f}")

# Importance des variables
importances = pd.DataFrame({
    'variable': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nImportance des variables (top 10):")
print(importances.head(10))

# ---- 5. Boosting AdaBoost ----
print("\n" + "=" * 60)
print("BOOSTING (AdaBoost avec stumps)")
print("=" * 60)

boost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=235),
    n_estimators=500,
    learning_rate=1.0,
    random_state=235
)
boost.fit(X_train, y_train)

y_score_boost = boost.predict_proba(X_test)[:, 1]
auc_boost = roc_auc_score(y_test, y_score_boost)
err_boost = 1 - accuracy_score(y_test, boost.predict(X_test))

print(f"AUC Boosting:         {auc_boost:.4f}")
print(f"Erreur Boosting:      {err_boost:.4f}")

# ---- 6. Synthèse ----
print("\n" + "=" * 60)
print("SYNTHÈSE DES RÉSULTATS")
print("=" * 60)

resultats = pd.DataFrame({
    'Méthode': ['CART', 'Bagging', 'Forêt aléatoire', 'Boosting'],
    'AUC':     [auc_tree, auc_bag, auc_rf, auc_boost],
    'Erreur':  [err_tree, err_bag, err_rf, err_boost]
})
print(resultats.round(4))

# ---- 7. Visualisation ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbes ROC
for name, y_score in [
    ('CART', y_score_tree),
    ('Bagging', y_score_bag),
    ('Random Forest', y_score_rf),
    ('Boosting', y_score_boost)
]:
    fpr, tpr, _ = roc_curve(y_test, y_score)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_score):.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_xlabel('Taux de faux positifs')
axes[0].set_ylabel('Taux de vrais positifs')
axes[0].set_title('Courbes ROC')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Importance des variables (top 10)
top10 = importances.head(10)
axes[1].barh(range(len(top10)), top10['importance'].values, color='steelblue')
axes[1].set_yticks(range(len(top10)))
axes[1].set_yticklabels(top10['variable'].values)
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 variables importantes (Random Forest)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('/Users/digitalnomad/Documents/students/stats-students/cyrille/STA211/samedi-06-juin/resultats.png', dpi=150)
plt.show()

print("\nFichier 'resultats.png' sauvegardé.")
