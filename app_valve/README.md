# Application web : prédiction de la condition de la valve

On entre un numéro de cycle et l'application indique si la valve est optimale (100 %) ou non, avec la probabilité, la vraie valeur et les signaux PS2 / FS1 du cycle.

## Installation

```bash
pip install -r requirements.txt
```

Placer les fichiers `PS2.txt`, `FS1.txt` et `profile.txt` dans le dossier `data/`.

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur (http://localhost:8501).

## Ré-entraîner le modèle (facultatif)

Le modèle est déjà fourni dans `model/modele.joblib`. Pour le recalculer :

```bash
python train.py
```

Le script reprend les étapes du notebook : 18 statistiques par cycle, sélection des 8 variables, découpage par blocs de conditions, Grid Search / Random Search, puis choix du modèle au meilleur F1 sur les « autres plis ». Il est entraîné sur les 2000 premiers cycles uniquement.

## Fichiers

- `app.py` : application Streamlit
- `features.py` : chargement des données et calcul des variables
- `train.py` : entraînement et sauvegarde du modèle
- `model/modele.joblib` : modèle final
