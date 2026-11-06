"""Baseline Random Forest vs PSO-tuned Random Forest. Saves results/results.json."""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data_prep import build_dataset
from optimizer import PSO

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "results.json")
SEED = 42

# name, low, high, is_integer
SPACE = [("n_estimators", 50, 150, True), ("max_depth", 3, 30, True),
         ("min_samples_split", 2, 20, True), ("min_samples_leaf", 1, 10, True),
         ("max_features", 0.1, 1.0, False)]
BOUNDS = [(lo, hi) for _, lo, hi, _ in SPACE]


def decode(x):
    """Turn a PSO position into RandomForest parameters."""
    return {n: (int(round(v)) if is_int else float(round(v, 3)))
            for (n, _, _, is_int), v in zip(SPACE, x)}


def make_rf(params=None):
    return RandomForestClassifier(random_state=SEED, n_jobs=1, **(params or {}))


def evaluate(model, X, y):
    pred, proba = model.predict(X), model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "test": {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
                 "recall": recall_score(y, pred), "f1": f1_score(y, pred),
                 "auc": roc_auc_score(y, proba)},
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def run(n_particles=10, n_iter=12, n_seeds=3, progress=None):
    X, y, info = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def cv_f1(params):
        return cross_val_score(make_rf(params), Xtr, ytr, cv=cv, scoring="f1", n_jobs=-1).mean()

    # 1) baseline: default hyperparameters
    base = make_rf().fit(Xtr, ytr)
    baseline = evaluate(base, Xte, yte)
    baseline["cv_f1"] = float(cv_f1(None))

    # 2) PSO, repeated with several random seeds
    cache, seeds, total = {}, [], n_seeds * n_iter

    def fitness(x):
        key = tuple(decode(x).values())
        if key not in cache:
            cache[key] = cv_f1(decode(x))
        return cache[key]

    for s in range(n_seeds):
        def cb(i, n, s=s):
            if progress:
                progress((s * n_iter + i) / total, f"Optimizing: seed {s + 1}/{n_seeds}, iteration {i}/{n}")
        best, fit, hist = PSO(fitness, BOUNDS, n_particles, n_iter, seed=s, callback=cb).run()
        params = decode(best)
        m = make_rf(params).fit(Xtr, ytr)
        seeds.append({"seed": s, "history": hist, "best_cv_f1": fit, "params": params,
                      "test_f1": float(f1_score(yte, m.predict(Xte)))})

    # 3) final model = best seed (chosen on CV score only, never on the test set)
    top = max(seeds, key=lambda r: r["best_cv_f1"])
    final = make_rf(top["params"]).fit(Xtr, ytr)
    optimized = evaluate(final, Xte, yte)
    optimized.update({"cv_f1": top["best_cv_f1"], "params": top["params"]})

    imp = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])[:15]
    results = {
        "info": {**info, "n_train": len(Xtr), "n_test": len(Xte)},
        "settings": {"n_particles": n_particles, "n_iter": n_iter, "n_seeds": n_seeds},
        "baseline": baseline, "optimized": optimized, "seeds": seeds,
        "importances": [{"feature": f, "importance": float(v)} for f, v in imp],
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    r = run(progress=lambda p, m: print(f"{p:5.0%}  {m}"))
    print("Baseline :", {k: round(v, 3) for k, v in r["baseline"]["test"].items()})
    print("Optimized:", {k: round(v, 3) for k, v in r["optimized"]["test"].items()})
    print("Best params:", r["optimized"]["params"])
