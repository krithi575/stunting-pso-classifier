"""Save the experiment results as a Markdown file.

Run after the experiment:   python export_report.py
Output: results/results.md
"""
import json
import os
from datetime import date

from experiment import RESULTS_PATH

OUT_PATH = os.path.join(os.path.dirname(RESULTS_PATH), "results.md")
METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
           ("f1", "F1"), ("auc", "ROC-AUC")]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit("No results found. Run `python experiment.py` first.")
    R = json.load(open(RESULTS_PATH))
    B, O, info, cfg = R["baseline"], R["optimized"], R["info"], R["settings"]

    rows = [[n, f"{B['test'][k]:.3f}", f"{O['test'][k]:.3f}", f"{O['test'][k] - B['test'][k]:+.3f}"]
            for k, n in METRICS]
    rows.append(["5-fold CV F1 (training)", f"{B['cv_f1']:.3f}", f"{O['cv_f1']:.3f}",
                 f"{O['cv_f1'] - B['cv_f1']:+.3f}"])

    def cm(m):
        (tn, fp), (fn, tp) = m["cm"]
        return md_table(["", "Predicted low", "Predicted high"],
                        [["**Actual low**", tn, fp], ["**Actual high**", fn, tp]])

    out = f"""# District child-stunting classification: results

Generated on {date.today().isoformat()}

Random Forest baseline vs Random Forest tuned with Particle Swarm Optimization (PSO), on NFHS-5 district data.

## Dataset
- Districts: {info['n_districts']} ({info['n_train']} for training, {info['n_test']} for testing)
- Indicators per district: {info['n_features']}
- Label: high risk when child stunting is above {info['threshold']}% ({info['n_high']} high, {info['n_low']} low)
- Removed from the inputs to prevent leakage: {len(info['dropped_leak_columns'])} child-nutrition outcome columns

## PSO settings
- Particles: {cfg['n_particles']}
- Iterations: {cfg['n_iter']}
- Independent runs (seeds): {cfg['n_seeds']}
- Fitness: mean stratified 5-fold cross-validated F1 on the training set

## Baseline vs tuned (held-out test set)
{md_table(["Metric", "Baseline", "PSO-tuned", "Change"], rows)}

## Best parameters found
{md_table(["Parameter", "Value"], [[k, v] for k, v in O['params'].items()])}

## Each PSO run
{md_table(["Run", "CV F1 (training)", "Test F1", "Parameters"],
          [[s['seed'] + 1, f"{s['best_cv_f1']:.3f}", f"{s['test_f1']:.3f}", s['params']] for s in R['seeds']])}

## Confusion matrices (test set)
**Baseline**

{cm(B)}

**PSO-tuned**

{cm(O)}

## Most important indicators (tuned model)
{md_table(["#", "Indicator", "Importance"],
          [[i + 1, f['feature'], f"{f['importance']:.4f}"] for i, f in enumerate(R['importances'])])}

## Notes
- The test set has only {info['n_test']} districts, so differences of a few points can be noise. Compare the repeated runs above.
- Feature importance shows association, not cause.
- Data: NFHS-5 district fact sheets (IIPS), compiled as CSV by Bhanu Pratap Vardhan, CC BY 4.0, https://doi.org/10.7910/DVN/42WNZF
"""
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
