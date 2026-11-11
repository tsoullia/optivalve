"""Ré-entraîne le modèle final du notebook et le sauvegarde dans model/modele.joblib.

Reprend les mêmes étapes que le notebook : variables, sélection (1.8), découpage par blocs,
Grid Search / Random Search (4.1, 4.2) et règle de choix du modèle final (4.4).
Usage : python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import N_DEV, charger_donnees, cible, construire_features

RANDOM_STATE = 0
SEUIL_CORR = 0.95
K_FEATURES = 8
SORTIE = Path(__file__).parent / "model" / "modele.joblib"


def selectionner_variables(X, y):
    """Sélection du notebook (1.8) : filtre de colinéarité puis rang moyen ANOVA / info mutuelle / forêt."""
    constantes = [c for c in X.columns if X[c].nunique() <= 2]
    corr = X.corr()
    corr_cible = X.apply(lambda col: col.corr(y)).dropna().sort_values(key=abs, ascending=False)
    gardees = []
    for feat in corr_cible.index:
        if feat in constantes:
            continue
        if all(abs(corr.loc[feat, g]) <= SEUIL_CORR for g in gardees):
            gardees.append(feat)
    f_scores, _ = f_classif(X[gardees], y)
    mi = mutual_info_classif(X[gardees], y, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(X[gardees], y)
    scores = pd.DataFrame({"F": f_scores, "mi": mi, "rf": rf.feature_importances_}, index=gardees)
    scores["rang"] = scores.rank(ascending=False).mean(axis=1)
    return scores.sort_values("rang").index[:K_FEATURES].tolist()


def main():
    print("Chargement des données...")
    ps2, fs1, profile = charger_donnees()
    features = construire_features(ps2, fs1)
    y = cible(profile)
    X_dev, X_test = features.iloc[:N_DEV], features.iloc[N_DEV:]
    y_dev, y_test = y.iloc[:N_DEV], y.iloc[N_DEV:]

    variables = selectionner_variables(X_dev, y_dev)
    print("Variables retenues :", variables)
    X_dev_sel, X_test_sel = X_dev[variables], X_test[variables]

    # Découpage par blocs de conditions (refroidisseur, fuite de pompe, accumulateur)
    groupes = profile.iloc[:N_DEV][["Cooler", "Pump_leakage", "Accumulator"]].astype(str).agg("-".join, axis=1).to_numpy()
    plis = list(StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X_dev_sel, y_dev, groupes))
    plis_bis = list(StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE + 1).split(X_dev_sel, y_dev, groupes))

    grilles = {
        "Régression logistique": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                                  {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
        "k-NN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                 {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15, 25],
                  "kneighborsclassifier__weights": ["uniform", "distance"]}),
        "Arbre de décision": (DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
                              {"max_depth": [2, 3, 4, 5, 8], "min_samples_leaf": [1, 5, 10, 20]}),
        "SVM RBF": (make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE)),
                    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}),
    }
    distributions = {
        "Forêt aléatoire": (RandomForestClassifier(random_state=RANDOM_STATE),
                            {"n_estimators": randint(100, 400), "max_depth": randint(3, 15),
                             "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2", 0.5]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"n_estimators": randint(50, 300), "learning_rate": loguniform(0.01, 0.3),
                               "max_depth": randint(1, 5), "subsample": uniform(0.6, 0.4)}),
    }

    recherches = {}
    for nom, (modele, grille) in grilles.items():
        recherches[nom] = GridSearchCV(modele, grille, scoring="f1", cv=plis, n_jobs=-1).fit(X_dev_sel, y_dev)
    for nom, (modele, distrib) in distributions.items():
        recherches[nom] = RandomizedSearchCV(modele, distrib, n_iter=20, scoring="f1", cv=plis, n_jobs=-1,
                                             random_state=RANDOM_STATE).fit(X_dev_sel, y_dev)

    # Règle du notebook (4.4) : meilleur F1 sur les « autres plis »
    autres = {nom: cross_val_score(r.best_estimator_, X_dev_sel, y_dev, cv=plis_bis, scoring="f1").mean()
              for nom, r in recherches.items()}
    for nom, s in sorted(autres.items(), key=lambda kv: -kv[1]):
        print(f"  {nom:22s} F1 CV (autres plis) = {s:.3f} | {recherches[nom].best_params_}")
    nom_final = max(autres, key=autres.get)
    final = recherches[nom_final].best_estimator_

    y_pred = final.predict(X_test_sel)
    print(f"\nModèle final : {nom_final}")
    print(f"Test final (205 cycles) : accuracy = {accuracy_score(y_test, y_pred):.3f} | F1 = {f1_score(y_test, y_pred):.3f}")

    SORTIE.parent.mkdir(exist_ok=True)
    joblib.dump({"modele": final, "variables": variables, "nom": nom_final,
                 "params": recherches[nom_final].best_params_}, SORTIE)
    print(f"Modèle sauvegardé dans {SORTIE}")


if __name__ == "__main__":
    main()
