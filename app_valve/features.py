"""Chargement des données et calcul des variables, identiques au notebook."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"
N_DEV = 2000  # 2000 premiers cycles : développement, le reste : test final
COLONNES_PROFILE = ["Cooler", "Valve", "Pump_leakage", "Accumulator", "Stable_flag"]


def charger_donnees(data_dir=DATA_DIR):
    ps2 = pd.read_csv(Path(data_dir) / "PS2.txt", sep=r"\s+", header=None)
    fs1 = pd.read_csv(Path(data_dir) / "FS1.txt", sep=r"\s+", header=None)
    profile = pd.read_csv(Path(data_dir) / "profile.txt", sep=r"\s+", header=None, names=COLONNES_PROFILE)
    assert len(ps2) == len(fs1) == len(profile), "Nombre de cycles incohérent entre les fichiers"
    return ps2, fs1, profile


def extraire_features(df, prefixe):
    """Résumé de chaque cycle par 9 statistiques descriptives (une ligne = un cycle)."""
    X = df.to_numpy(dtype=float)
    f = pd.DataFrame(index=df.index)
    f[f"{prefixe}_mean"] = X.mean(axis=1)
    f[f"{prefixe}_std"] = X.std(axis=1)
    f[f"{prefixe}_min"] = X.min(axis=1)
    f[f"{prefixe}_max"] = X.max(axis=1)
    f[f"{prefixe}_median"] = np.median(X, axis=1)
    f[f"{prefixe}_q25"] = np.percentile(X, 25, axis=1)
    f[f"{prefixe}_q75"] = np.percentile(X, 75, axis=1)
    f[f"{prefixe}_skew"] = stats.skew(X, axis=1)
    f[f"{prefixe}_kurtosis"] = stats.kurtosis(X, axis=1)
    return f


def construire_features(ps2, fs1):
    return pd.concat([extraire_features(ps2, "PS2"), extraire_features(fs1, "FS1")], axis=1)


def cible(profile):
    """1 = valve optimale (100 %), 0 = non optimale."""
    return (profile["Valve"] == 100).astype(int)
