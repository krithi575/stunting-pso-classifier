"""Load NFHS-5 district data and build a clean, model-ready table."""
import os
import re
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "NFHS-5-Districts.csv")
URL = "https://raw.githubusercontent.com/pratapvardhan/NFHS-5/master/NFHS-5-Districts.csv"

NATIONAL_STUNTING = 35.5  # % of children under 5 (NFHS-5, India)
# Child nutrition outcomes are removed from the features so the model cannot "cheat".
LEAK_KEYS = ["stunted", "wasted", "underweight", "overweight"]


def load_raw():
    """Read the CSV from data/ (downloads it once if missing)."""
    if not os.path.exists(PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    return pd.read_csv(PATH)


def build_dataset(max_missing=0.30, return_meta=False):
    """Return X (features), y (1 = high stunting), and a dict of dataset facts.
    With return_meta=True also returns a table of State, District and actual stunting %."""
    df = load_raw()
    # tidy indicator names: drop leading "73. " and footnote digits like ")18"
    df["Indicator"] = (df["Indicator"].str.replace(r"^\d+\.\s*", "", regex=True)
                                      .str.replace(r"\)\d+", ")", regex=True))
    # long format -> one row per district, one column per indicator
    wide = df.pivot_table(index=["State", "District"], columns="Indicator",
                          values="NFHS-5", aggfunc="first")

    target = [c for c in wide.columns if "children under 5" in c.lower() and "stunted" in c.lower()][0]
    wide = wide.dropna(subset=[target])
    stunting = wide[target]

    leak = [c for c in wide.columns
            if "children under 5" in c.lower() and any(k in c.lower() for k in LEAK_KEYS)]
    X = wide.drop(columns=leak)

    X = X.loc[:, X.isna().mean() <= max_missing]   # drop columns with too many gaps
    X = X.fillna(X.median())                        # median imputation
    X = X.loc[:, X.nunique() > 1]                   # drop constant columns

    threshold = NATIONAL_STUNTING
    share = (stunting > threshold).mean()
    if share < 0.25 or share > 0.75:                # keep classes reasonably balanced
        threshold = float(stunting.median())
    y = (stunting > threshold).astype(int)

    info = {
        "n_districts": int(len(X)),
        "n_features": int(X.shape[1]),
        "threshold": round(float(threshold), 1),
        "n_high": int(y.sum()),
        "n_low": int((1 - y).sum()),
        "dropped_leak_columns": leak,
    }
    meta = X.index.to_frame(index=False)
    meta["stunting_pct"] = stunting.values
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    return (X, y, info, meta) if return_meta else (X, y, info)


if __name__ == "__main__":
    X, y, info = build_dataset()
    print(info)
