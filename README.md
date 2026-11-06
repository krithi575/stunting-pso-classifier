# District child-stunting risk: Random Forest tuned with PSO

Classifies Indian districts as high or low stunting risk from NFHS-5 indicators, then tunes the
Random Forest with Particle Swarm Optimization (written from scratch) and compares it to the default model.
Data: NFHS-5 district fact sheets by Bhanu Pratap Vardhan (CC BY 4.0), https://doi.org/10.7910/DVN/42WNZF
The CSV is already included in `data/`, so no download or login is needed.

## Run it in VS Code

1. Install Python 3.10 or newer (python.org). Install VS Code and its Python extension.
2. In VS Code: File > Open Folder, and choose this project folder.
3. Open a terminal: Terminal > New Terminal.
4. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
5. Install the packages: `pip install -r requirements.txt`
6. Start the dashboard: `streamlit run app.py`
7. Your browser opens. Press **Run experiment** in the sidebar and wait a few minutes. The charts appear when it finishes.

Prefer the terminal? Run `python experiment.py` first (prints the metrics and saves `results/results.json`),
then `streamlit run app.py` to see the charts.

## Files

| File | What it does |
|---|---|
| `data_prep.py` | Loads the CSV, pivots it to one row per district, cleans it, removes child-nutrition outcome columns, creates the high/low stunting label |
| `optimizer.py` | PSO: swarm, velocity update, position update, personal and global best |
| `experiment.py` | Baseline model, PSO runs (several seeds), final evaluation, saves results |
| `app.py` | The dashboard (all charts) |
| `export_report.py` | Saves the results as `results/results.md` (run `python export_report.py` after the experiment) |

## For your report (rubric mapping)

- Dataset and preprocessing: `data_prep.py`
- Model: Random Forest baseline in `experiment.py`
- Optimization: `optimizer.py`; the search space is `SPACE` in `experiment.py`; the fitness is 5-fold CV F1 on the training set only
- Results and plots: the dashboard tabs (use the camera icon on any chart to save it as an image)

## Tips

- Quick test: set Particles 5, Iterations 4, Repeated runs 2 in the sidebar.
- If results change slightly between machines, that is normal for a small dataset. Report the repeated-run table.
- Change the label threshold in `data_prep.py` (`NATIONAL_STUNTING`) if you want a different definition of high risk.
