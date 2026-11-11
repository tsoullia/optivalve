"""Application web : prédiction de la condition de la valve pour un numéro de cycle.
Lancement : streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import N_DEV, charger_donnees, cible, construire_features

MODELE = Path(__file__).parent / "model" / "modele.joblib"

st.set_page_config(page_title="Condition de la valve", page_icon="⚙️", layout="centered")


@st.cache_resource
def charger_modele():
    return joblib.load(MODELE)


@st.cache_data
def charger():
    ps2, fs1, profile = charger_donnees()
    return ps2, fs1, profile, construire_features(ps2, fs1), cible(profile)


if not MODELE.exists():
    st.error("Modèle introuvable. Lance d'abord `python train.py`.")
    st.stop()

paquet = charger_modele()
modele, variables = paquet["modele"], paquet["variables"]

with st.spinner("Chargement des données..."):
    ps2, fs1, profile, features, y = charger()
n_cycles = len(profile)

st.title("Maintenance prédictive : condition de la valve")
st.write(
    f"Entrez un numéro de cycle (1 à {n_cycles}). Le modèle prédit si la valve est **optimale** (100 %) "
    f"ou **non optimale**. Les cycles 1 à {N_DEV} ont servi à construire le modèle, "
    f"les cycles {N_DEV + 1} à {n_cycles} forment le test final."
)

numero = st.number_input("Numéro de cycle", min_value=1, max_value=n_cycles, value=N_DEV + 1, step=1)
i = int(numero) - 1  # index pandas (commence à 0)

x = features.iloc[[i]][variables]
prediction = int(modele.predict(x)[0])
proba = float(modele.predict_proba(x)[0, 1]) if hasattr(modele, "predict_proba") else None
reel = int(y.iloc[i])

col1, col2, col3 = st.columns(3)
col1.metric("Prédiction", "Optimale" if prediction == 1 else "Non optimale")
if proba is not None:
    col2.metric("Probabilité « optimale »", f"{proba:.1%}")
col3.metric("Valeur réelle", f"{int(profile['Valve'].iloc[i])} %")

if prediction == reel:
    st.success("Prédiction correcte.")
else:
    st.error("Prédiction incorrecte.")

st.caption(
    ("Cycle de **test final** : jamais vu par le modèle." if i >= N_DEV
     else "Cycle de **développement** : il a servi à l'entraînement, la prédiction est donc moins informative.")
    + f" Modèle : {paquet['nom']}."
)

st.subheader("Signaux du cycle")
temps_ps2 = np.arange(ps2.shape[1]) / 100  # 100 Hz
temps_fs1 = np.arange(fs1.shape[1]) / 10   # 10 Hz
st.write("Pression PS2 (bar)")
st.line_chart(pd.DataFrame({"PS2": ps2.iloc[i].to_numpy()}, index=pd.Index(temps_ps2, name="Temps (s)")), height=220)
st.write("Débit FS1 (l/min)")
st.line_chart(pd.DataFrame({"FS1": fs1.iloc[i].to_numpy()}, index=pd.Index(temps_fs1, name="Temps (s)")), height=220)

with st.expander("Variables utilisées par le modèle"):
    st.dataframe(x.T.rename(columns={i: "Valeur"}).round(4), width="stretch")

with st.expander("Conditions de fonctionnement du cycle (profile.txt)"):
    st.dataframe(profile.iloc[[i]].reset_index(drop=True), width="stretch", hide_index=True)
