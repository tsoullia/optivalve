OptiValve – Maintenance prédictive d'une valve hydraulique

Projet de Machine Learning (ING2 BDML, Efrei, groupe 3) : prédire, pour chaque cycle d'un banc d'essai hydraulique, si la valve est dans son état optimal (100 %) ou dégradée, à partir de deux capteurs (pression PS2 et débit FS1).

Résultat : le modèle final, un Gradient Boosting, classe correctement les 205 cycles du test final (accuracy, précision, rappel, F1 et AUC de 1,000).

Données

Source : UCI – Condition monitoring of hydraulic systems.

Les fichiers utilisés se trouvent dans data/ :

PS2.txt : pression, 100 Hz, soit 6 000 mesures par cycle
FS1.txt : débit, 10 Hz, soit 600 mesures par cycle
profile.txt : état réel des composants (la 2e colonne donne l'état de la valve)

Les 2 000 premiers cycles servent au développement. Les 205 suivants forment le test final, utilisé une seule fois.

Structure du dépôt
```
├── data/              Données (PS2, FS1, profile)
├── app_valve/         Application web (bonus)
├── fichier.ipynb      Notebook principal du projet
├── cours.ipynb        Notebook du TP de cours (base de départ)
├── requirements.txt   Librairies nécessaires
└── README.md
```
Démarche
Prétraitement : chaque cycle est résumé par 18 statistiques ; sélection de 8 variables (filtre de colinéarité, puis ANOVA, information mutuelle et forêt aléatoire).
Découpage par blocs de conditions (StratifiedGroupKFold), pour éviter la fuite de données entre cycles presque identiques.
8 modèles comparés : régression logistique, k-NN, arbre de décision, Naive Bayes, SVM linéaire, SVM RBF, forêt aléatoire, Gradient Boosting.
Optimisation par Grid Search et Random Search, puis choix du modèle final selon une règle fixée à l'avance.
Évaluation unique sur le test final, suivie de contrôles de cohérence (fuite de données, stabilité, robustesse).
Lancer le notebook
Ouvrir fichier.ipynb, puis exécuter toutes les cellules (Restart puis Run All).

Application web (bonus)

Une application Streamlit donne la prédiction pour un numéro de cycle saisi en entrée. Elle affiche aussi la probabilité, la vraie condition de la valve et les courbes du cycle.

```
cd app_valve
pip install -r requirements.txt
streamlit run app.py
```

Il faut d'abord copier PS2.txt, FS1.txt et profile.txt dans app_valve/data/. Les détails sont dans app_valve/README.md.

Équipe

Thomas Soulliaert · Shreyas Venkatesan · Andréa Dario
