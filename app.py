"""Beige results dashboard.  Run with:  streamlit run app.py"""
import json
import os

import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from data_prep import build_dataset
from experiment import RESULTS_PATH, SEED, make_rf, run

BG, SURFACE, INK, MUTED, LINE = "#EFE8DA", "#F8F4EA", "#2E2A24", "#6F675A", "#D9CFBB"
SAND, OLIVE = "#B9A98B", "#5B6B4E"

st.set_page_config(page_title="District stunting risk", layout="wide")

# Every text element is forced to dark ink so it stays readable even if the
# browser or Streamlit is in dark mode.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;800&display=swap');
:root, html, body, .stApp {{ color-scheme: light; }}
html, body, .stApp {{ font-family: 'Manrope', sans-serif; }}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp td, .stApp th, .stApp button p, .stApp input {{ font-family: 'Manrope', sans-serif; }}
[data-testid="stToolbar"], [data-testid="stDeployButton"] {{ display: none !important; }}
.stApp {{ background: {BG} !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1100px; padding: 2.5rem 2rem 4rem; }}

/* text: dark ink everywhere */
.stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp div[data-testid="stMarkdownContainer"], .stApp small,
.stApp [data-testid="stWidgetLabel"] *, .stApp [data-testid="stCaptionContainer"] * {{ color: {INK} !important; }}
.stApp .muted, .stApp .muted * {{ color: {MUTED} !important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {SURFACE} !important; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] * {{ color: {INK} !important; }}
[data-testid="stNumberInputContainer"], [data-testid="stNumberInputContainer"] > div,
[data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="base-input"] {{
    background: {BG} !important; border-color: {LINE} !important; border-radius: 10px; }}
[data-testid="stNumberInput"] input {{ background: {BG} !important; color: {INK} !important;
    -webkit-text-fill-color: {INK} !important; font-weight: 600; }}
[data-testid="stNumberInput"] button {{ background: {BG} !important; border-color: {LINE} !important; }}
[data-testid="stNumberInput"] button svg {{ fill: {INK} !important; color: {INK} !important; }}
.stButton > button {{ background: {OLIVE} !important; border: 0 !important; border-radius: 10px;
    font-weight: 600; width: 100%; padding: .55rem 1rem; }}
.stButton > button, .stButton > button * {{ color: #FFFFFF !important; }}
.stButton > button:hover {{ background: #4A5940 !important; }}
[data-testid="stProgress"] > div > div {{ background: {LINE} !important; }}
[data-testid="stProgress"] > div > div > div {{ background: {OLIVE} !important; }}
[data-testid="stAlert"] {{ background: {SURFACE} !important; border: 1px solid {LINE}; border-radius: 12px; }}

/* dropdowns */
[data-baseweb="select"] > div {{ background: {BG} !important; border-color: {LINE} !important; border-radius: 10px; }}
[data-baseweb="select"] *, [data-baseweb="select"] svg {{ color: {INK} !important; fill: {INK} !important; }}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="menu"] ul,
[data-baseweb="menu"] li {{ background: {SURFACE} !important; color: {INK} !important; }}
[data-baseweb="menu"] li:hover, [data-baseweb="menu"] li[aria-selected="true"] {{ background: {BG} !important; }}
[data-baseweb="menu"] li * {{ color: {INK} !important; }}
[data-testid="stCheckbox"] label span {{ color: {INK} !important; }}

/* prediction bar */
.bar {{ position: relative; height: 12px; background: {LINE}; border-radius: 99px; margin: 10px 0 6px; }}
.bar .fill {{ height: 100%; border-radius: 99px; }}
.bar .mark {{ position: absolute; top: -4px; width: 2px; height: 20px; background: {INK}; }}

/* title */
.title {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 .5rem; }}
.sub {{ font-size: 1.05rem; max-width: 620px; margin-bottom: 2rem; line-height: 1.5; }}

/* cards */
.card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.card .l {{ font-size: .85rem; font-weight: 500; }}
.card .v {{ font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: 2px 0 0; }}
.card .d {{ font-size: .8rem; font-weight: 600; margin-top: 2px; }}
.card .d.pos {{ color: {OLIVE} !important; }}
.card .d.neg {{ color: #8A5A44 !important; }}
.card .d.zero {{ color: {MUTED} !important; }}
.gap {{ height: 22px; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ background: transparent !important; padding: 10px 16px; }}
.stTabs [data-baseweb="tab"] p {{ color: {MUTED} !important; font-weight: 600; }}
.stTabs [aria-selected="true"] p {{ color: {INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {OLIVE} !important; }}
.stTabs [data-baseweb="tab-border"] {{ background: {LINE} !important; }}

/* tables */
table.tbl {{ width: 100%; border-collapse: collapse; background: {SURFACE}; border: 1px solid {LINE};
    border-radius: 12px; overflow: hidden; font-size: .93rem; }}
table.tbl th {{ text-align: left; font-weight: 600; padding: 10px 14px; border-bottom: 1px solid {LINE};
    color: {MUTED} !important; background: {SURFACE}; }}
table.tbl td {{ padding: 10px 14px; border-bottom: 1px solid {LINE}; color: {INK} !important; }}
table.tbl tr:last-child td {{ border-bottom: 0; }}
.h {{ font-size: 1.05rem; font-weight: 700; margin: 0 0 .6rem; }}
.note {{ font-size: .92rem; max-width: 700px; line-height: 1.55; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope, sans-serif", color=INK, size=13),
                      margin=dict(l=20, r=20, t=20, b=20),
                      legend=dict(orientation="h", y=-0.22, x=0, font=dict(color=INK)))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=INK, size=12),
                     title_font=dict(color=INK), automargin=True, title_standoff=14)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=INK, size=12),
                     title_font=dict(color=INK), automargin=True, title_standoff=14)
    return fig


def show(fig, h=380):
    st.plotly_chart(style(fig, h), theme=None, width="stretch")


def card(label, value, delta=None, fmt="{:.3f}"):
    d = ""
    if delta is not None:
        cls = "zero" if abs(delta) < 0.0005 else ("pos" if delta > 0 else "neg")
        d = f'<div class="d {cls}">{delta:+.3f} vs baseline</div>'
    return f'<div class="card"><div class="l muted">{label}</div><div class="v">{fmt.format(value)}</div>{d}</div>'


def table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Run experiment")
    particles = st.number_input("Particles", 4, 40, 10)
    iters = st.number_input("Iterations", 3, 60, 12)
    seeds = st.number_input("Repeated runs", 1, 10, 3)
    if st.button("Run experiment"):
        bar = st.progress(0.0, text="Preparing data")
        run(int(particles), int(iters), int(seeds), progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
        st.rerun()
    st.markdown('<p class="muted" style="font-size:.85rem">A full run takes a few minutes. '
                'Results are saved and reload instantly next time.</p>', unsafe_allow_html=True)

st.markdown('<div class="title">Which districts are most at risk<br>of child stunting?</div>'
            '<div class="sub muted">A Random Forest classifier on NFHS-5 district data, '
            'tuned with particle swarm optimization.</div>', unsafe_allow_html=True)

if not os.path.exists(RESULTS_PATH):
    st.info("No results yet. Press Run experiment in the sidebar, or run `python experiment.py` in the terminal.")
    st.stop()

R = json.load(open(RESULTS_PATH))
B, O, info = R["baseline"], R["optimized"], R["info"]
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

st.markdown('<div class="muted" style="margin-bottom:.5rem;font-weight:600">Tuned model on unseen districts</div>',
            unsafe_allow_html=True)
cols = st.columns(5, gap="small")
for c, m, n in zip(cols, metrics, names):
    c.markdown(card(n, O["test"][m], O["test"][m] - B["test"][m]), unsafe_allow_html=True)
st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

t1, tp, t2, t3, t4, t5 = st.tabs(["Overview", "District predictor", "Convergence", "Baseline vs tuned", "Diagnostics", "Key indicators"])

with t1:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    c = st.columns(4, gap="small")
    c[0].markdown(card("Districts", info["n_districts"], fmt="{:,}"), unsafe_allow_html=True)
    c[1].markdown(card("Indicators per district", info["n_features"], fmt="{:,}"), unsafe_allow_html=True)
    c[2].markdown(card(f"High risk (above {info['threshold']}%)", info["n_high"], fmt="{:,}"), unsafe_allow_html=True)
    c[3].markdown(card("Low risk", info["n_low"], fmt="{:,}"), unsafe_allow_html=True)
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown('<div class="h">How it was tested</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="note">A district is high risk when child stunting is above {info["threshold"]}%. '
                    f'The model learned from {info["n_train"]} districts and was scored on {info["n_test"]} it had '
                    f'never seen. Child nutrition outcomes were removed from the inputs, so the model cannot read '
                    f'the answer directly.</p>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="h">Best settings found</div>', unsafe_allow_html=True)
        st.markdown(table(["Parameter", "Value"], [[k, v] for k, v in O["params"].items()]),
                    unsafe_allow_html=True)


def card_text(label, text, sub=""):
    return (f'<div class="card"><div class="l muted">{label}</div>'
            f'<div class="v" style="font-size:1.6rem">{text}</div>'
            f'<div class="d zero">{sub}</div></div>')


@st.cache_resource(show_spinner="Loading the tuned model")
def load_model(params_key):
    """Rebuild the tuned model exactly as experiment.py did (same split, same seed)."""
    X, y, _, meta = build_dataset(return_meta=True)
    Xtr, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model = make_rf(json.loads(params_key)).fit(Xtr, y.loc[Xtr.index])
    return model, X, y, meta, set(Xtr.index)


with tp:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    model, Xall, yall, meta, train_idx = load_model(json.dumps(O["params"], sort_keys=True))
    only_new = st.checkbox("Only show districts the model has never seen (recommended)", value=True)
    pool = meta[~meta.index.isin(train_idx)] if only_new else meta
    c1, c2 = st.columns(2, gap="large")
    state = c1.selectbox("State", sorted(pool["State"].unique()))
    options = pool[pool["State"] == state].sort_values("District")
    district = c2.selectbox("District", options["District"].tolist())
    i = int(options[options["District"] == district].index[0])

    proba = float(model.predict_proba(Xall.loc[[i]])[0, 1])
    pred_high = proba >= 0.5
    actual_high = bool(yall.loc[i])
    actual_pct = float(meta.loc[i, "stunting_pct"])
    seen = i in train_idx

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3, gap="small")
    k1.markdown(card_text("Model prediction", "High risk" if pred_high else "Low risk",
                          f"{proba:.0%} chance of high risk"), unsafe_allow_html=True)
    k2.markdown(card_text("Actual stunting rate", f"{actual_pct:.1f}%",
                          f"{'High' if actual_high else 'Low'} risk (cut-off {info['threshold']}%)"),
                unsafe_allow_html=True)
    k3.markdown(card_text("Result", "Correct" if pred_high == actual_high else "Incorrect",
                          "Seen during training" if seen else "Held-out district, never seen"),
                unsafe_allow_html=True)

    color = "#8A5A44" if pred_high else OLIVE
    st.markdown(f'<div class="bar"><div class="fill" style="width:{proba * 100:.0f}%;background:{color}"></div>'
                f'<div class="mark" style="left:50%"></div></div>'
                f'<div class="muted" style="font-size:.82rem">Probability of high risk. The line marks 50%, '
                f'the point where the prediction switches to high risk.</div>', unsafe_allow_html=True)

    st.markdown('<div class="h" style="margin-top:1.6rem">What the model looked at</div>', unsafe_allow_html=True)
    top = [f["feature"] for f in R["importances"][:6]]
    rows = [[f, f"{Xall.loc[i, f]:.1f}", f"{Xall[f].median():.1f}"] for f in top if f in Xall.columns]
    st.markdown(table(["Most important indicators", district, "All-district median"], rows),
                unsafe_allow_html=True)
    st.markdown('<p class="note muted" style="margin-top:.8rem">Districts used in training are predicted almost '
                'perfectly, so judge the model on held-out districts. The indicators show association, '
                'not cause.</p>', unsafe_allow_html=True)

with t2:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    fig = go.Figure()
    shades = ["#5B6B4E", "#7E8E6F", "#9FAE90", "#B7C2A8", "#CBD3BF"]
    for i, s in enumerate(R["seeds"]):
        fig.add_scatter(y=s["history"], mode="lines+markers", name=f"Run {s['seed'] + 1}",
                        line=dict(color=shades[i % 5], width=2.5), marker=dict(size=5))
    fig.add_hline(y=B["cv_f1"], line_dash="dash", line_color=SAND,
                  annotation_text="Baseline", annotation_font_color=MUTED)
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Best cross-validated F1")
    fig.update_yaxes(tickformat=".3f")
    fig.update_xaxes(dtick=2)
    show(fig, 420)
    st.markdown('<p class="note muted">Each line is one independent swarm. A line that rises and then flattens '
                'shows the swarm settling on good parameters. Several runs show whether the result is stable.</p>',
                unsafe_allow_html=True)

with t3:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(x=names, y=[B["test"][m] for m in metrics], name="Baseline", marker_color=SAND)
    fig.add_bar(x=names, y=[O["test"][m] for m in metrics], name="PSO-tuned", marker_color=OLIVE)
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(range=[0, 1])
    show(fig, 360)
    st.markdown('<div class="h" style="margin-top:1rem">Each repeated run</div>', unsafe_allow_html=True)
    st.markdown(table(["Run", "CV F1 (training)", "Test F1", "n_estimators", "max_depth"],
                      [[s["seed"] + 1, f"{s['best_cv_f1']:.3f}", f"{s['test_f1']:.3f}",
                        s["params"]["n_estimators"], s["params"]["max_depth"]] for s in R["seeds"]]),
                unsafe_allow_html=True)
    st.markdown(f'<p class="note muted" style="margin-top:.8rem">Cross-validated F1: baseline {B["cv_f1"]:.3f}, '
                f'tuned {O["cv_f1"]:.3f}. The test set is small, so differences of a few points can be noise.</p>',
                unsafe_allow_html=True)

with t4:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    for col, title, M, color in [(a, "Baseline", B, SAND), (b, "PSO-tuned", O, OLIVE)]:
        with col:
            st.markdown(f'<div class="h">{title}</div>', unsafe_allow_html=True)
            cm = M["cm"]
            fig = go.Figure(go.Heatmap(z=cm, x=["Low risk", "High risk"], y=["Low risk", "High risk"],
                                       colorscale=[[0, SURFACE], [1, color]], showscale=False,
                                       text=cm, texttemplate="%{text}", textfont=dict(size=24, color=INK)))
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            fig.update_yaxes(autorange="reversed")
            show(fig, 300)
    st.markdown('<div class="h" style="margin-top:1rem">ROC curve</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_scatter(x=B["roc"]["fpr"], y=B["roc"]["tpr"], name=f"Baseline (AUC {B['test']['auc']:.2f})",
                    line=dict(color=SAND, width=3))
    fig.add_scatter(x=O["roc"]["fpr"], y=O["roc"]["tpr"], name=f"PSO-tuned (AUC {O['test']['auc']:.2f})",
                    line=dict(color=OLIVE, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=LINE, dash="dash"), showlegend=False)
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    show(fig, 400)

with t5:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    imp = R["importances"][::-1]
    fig = go.Figure(go.Bar(x=[i["importance"] for i in imp], y=[(i["feature"][:48] + "...") if len(i["feature"]) > 48 else i["feature"] for i in imp],
                           orientation="h", marker_color=OLIVE))
    fig.update_layout(xaxis_title="Importance in the tuned model")
    fig.update_yaxes(tickfont=dict(size=11, color=INK))
    show(fig, 600)
    st.markdown('<p class="note muted">These indicators matter most for separating high-risk from low-risk '
                'districts. They show association, not cause.</p>', unsafe_allow_html=True) 