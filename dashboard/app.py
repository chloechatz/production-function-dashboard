import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

st.set_page_config(page_title="Production Function Time Machine", layout="wide")

MODELS_DIR = Path(__file__).parent / "models"

FEATURES = ["revenue_eur", "annual_wage_eur", "squad_value_m", "net_transfer_spend_m", "average_squad_age", "ucl_encoded"]
FEATURE_LABELS = {
    "revenue_eur": "Annual Revenue (€ millions)",
    "annual_wage_eur": "Annual Wage Bill (€ millions)",
    "squad_value_m": "Squad Market Value (€ millions)",
    "net_transfer_spend_m": "Net Transfer Spend (€ millions)",
    "average_squad_age": "Average Squad Age (years)",
    "ucl_encoded": "Champions League Depth",
}
ERA_LABELS = {"pre_ffp": "Pre-FFP (2010-2013)", "post_ffp": "Post-FFP (2013-2020)", "post_covid": "Post-COVID (2020-2025)"}

UCL_STAGES = ["Did Not Qualify", "Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Runner-up", "Winner"]

# Slider ranges match the min/max in the dataset.

BOUNDS = {
    "revenue_m": (30, 1120), "wage_m": (15, 600), "squad_value_m": (48, 1460),
    "net_spend_m": (-565, 225), "squad_age": (22.0, 28.7),
}
DEFAULTS = {"revenue_m": 200, "wage_m": 100, "squad_value_m": 300, "net_spend_m": 0, "squad_age": 25.5, "ucl_depth": UCL_STAGES[0]}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


@st.cache_resource
def load_era(era_slug):
    # cache models and explainers to avoid reloading
    model = joblib.load(MODELS_DIR / f"model_{era_slug}.pkl")
    scaler = joblib.load(MODELS_DIR / f"scaler_{era_slug}.pkl")
    explainer = joblib.load(MODELS_DIR / f"explainer_{era_slug}.pkl")
    return model, scaler, explainer


@st.cache_data
def load_real_clubs():
    # check possible locations for the dataset
    possible_paths = [
        Path(__file__).resolve().parent.parent / "master_panel_clean.csv",
        Path.cwd() / "master_panel_clean.csv",
    ]
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        st.error(f"Can't find master_panel_clean.csv. Looked in: {possible_paths}")
        st.stop()

    df = pd.read_csv(data_path)
    leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
    df = df[df['league'].isin(leagues)].copy()
    df['label'] = df['club'] + " — " + df['season'] + " (" + df['league'] + ")"
    return df


df_real = load_real_clubs()

st.title("The Production Function Time Machine")
st.caption("Enter a club's financial profile, and see how the model's prediction changes across football's regulatory eras.")

# Option to load a real club-season data
st.sidebar.header("Start from a Real Club (optional)")
club_options = ["-- Custom / manual input --"] + sorted(df_real['label'].unique().tolist())
selected_label = st.sidebar.selectbox("Pick a real club-season", club_options)

if selected_label != "-- Custom / manual input --":
    if st.sidebar.button("Load this profile into the sliders"):
        row = df_real[df_real['label'] == selected_label].iloc[0]

        def safe_millions(raw_eur, fallback_millions):
            # convert euros to millions
            if raw_eur is None or pd.isna(raw_eur):
                return fallback_millions
            return round(float(raw_eur) / 1_000_000, 1)

        def safe_float(raw, fallback):
            # use default value if data is missing
            if raw is None or pd.isna(raw):
                return fallback
            return round(float(raw), 1)

        def safe_int(raw, fallback):
            if raw is None or pd.isna(raw):
                return fallback
            return int(raw)

        st.session_state['revenue_m'] = safe_millions(row['revenue_eur'], DEFAULTS['revenue_m'])
        st.session_state['wage_m'] = safe_millions(row['annual_wage_eur'], DEFAULTS['wage_m'])
        st.session_state['squad_value_m'] = safe_float(row['squad_value_m'], DEFAULTS['squad_value_m'])
        st.session_state['net_spend_m'] = safe_float(row['net_transfer_spend_m'], DEFAULTS['net_spend_m'])
        st.session_state['squad_age'] = safe_float(row['average_squad_age'], DEFAULTS['squad_age'])
        
        # matching label string, not the raw number.
        default_ucl_idx = UCL_STAGES.index(DEFAULTS['ucl_depth'])
        ucl_idx = safe_int(row['ucl_encoded'], default_ucl_idx)
        ucl_idx = max(0, min(ucl_idx, len(UCL_STAGES) - 1))
        st.session_state['ucl_depth'] = UCL_STAGES[ucl_idx]

        missing_fields = [
            label for label, raw in [
                ("Revenue", row['revenue_eur']), ("Wage Bill", row['annual_wage_eur']),
                ("Squad Value", row['squad_value_m']), ("Net Transfer Spend", row['net_transfer_spend_m']),
                ("Squad Age", row['average_squad_age']), ("Champions League Depth", row['ucl_encoded']),
            ] if pd.isna(raw)
        ]
        if missing_fields:
            st.sidebar.warning(
                f"This club-season has no recorded value for: {', '.join(missing_fields)}. "
                "Those sliders were reset to their default instead."
            )
        st.rerun()

# Manual sliders 
st.sidebar.header("Club Financial Profile")
revenue_m = st.sidebar.slider("Annual Revenue (€ millions)", *BOUNDS["revenue_m"], step=10, key="revenue_m")
wage_m = st.sidebar.slider("Annual Wage Bill (€ millions)", *BOUNDS["wage_m"], step=5, key="wage_m")
squad_value_m = st.sidebar.slider("Squad Market Value (€ millions)", *BOUNDS["squad_value_m"], step=10, key="squad_value_m")
net_spend_m = st.sidebar.slider("Net Transfer Spend (€ millions)", *BOUNDS["net_spend_m"], step=5, key="net_spend_m")
squad_age = st.sidebar.slider("Average Squad Age (years)", *BOUNDS["squad_age"], step=0.1, key="squad_age")
ucl_stage_selected = st.sidebar.select_slider("Champions League Depth", options=UCL_STAGES, key="ucl_depth")
ucl_depth = UCL_STAGES.index(ucl_stage_selected)  # convert the label back to the 0-6 index the model expects

st.sidebar.caption(
    "Sliders are bounded to the actual range observed across all 28 clubs and 15 seasons in this study. "
    "Predictions get less reliable the further a combination sits from a typical club profile, "
    "since the model has seen few or no similar examples during training."
)

input_row = pd.DataFrame([{
    "revenue_eur": revenue_m * 1_000_000,
    "annual_wage_eur": wage_m * 1_000_000,
    "squad_value_m": squad_value_m,
    "net_transfer_spend_m": net_spend_m,
    "average_squad_age": squad_age,
    "ucl_encoded": ucl_depth,
}])[FEATURES]

# eras to compare 
st.subheader("Compare Two Eras")
col_select_a, col_select_b = st.columns(2)
with col_select_a:
    era_a = st.selectbox("Panel A — Era", options=list(ERA_LABELS.keys()), format_func=lambda x: ERA_LABELS[x], index=0)
with col_select_b:
    era_b = st.selectbox("Panel B — Era", options=list(ERA_LABELS.keys()), format_func=lambda x: ERA_LABELS[x], index=2)


def predict_for_era(era_slug, input_row):
    model, scaler, explainer = load_era(era_slug)
    X_scaled = scaler.transform(input_row)
    return model.predict(X_scaled)[0]


pred_a = predict_for_era(era_a, input_row)
pred_b = predict_for_era(era_b, input_row)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"### {ERA_LABELS[era_a]}")
    st.metric("Predicted Standing (0=worst, 1=best)", f"{pred_a:.2f}")
with col_b:
    st.markdown(f"### {ERA_LABELS[era_b]}")
    st.metric("Predicted Standing (0=worst, 1=best)", f"{pred_b:.2f}")

diff = pred_b - pred_a
direction = "better" if diff > 0 else "worse" if diff < 0 else "the same"
st.info(
    f"With the exact same financial profile, this club's predicted standing would be "
    f"**{direction}** in the {ERA_LABELS[era_b]} era compared to the {ERA_LABELS[era_a]} era "
    f"(a shift of {diff:+.2f} on the 0–1 scale)."
)
st.caption(
    "Each era panel is a separately trained model with its own baseline, not one model "
    "applied to two time periods — this comparison shows how the *learned relationship* "
    "differs by era, not a single model's output changing over time."
)

# SHAP explanation
st.subheader("Why the Model Predicted This")
st.caption(
    "Each bar shows how much that factor pushed the prediction away from the model's average expected value. "
    "Red bars (positive) push toward a better standing; blue bars (negative) push toward a worse one. "
    "The axis is the same 0–1 standing scale, zoomed in around this prediction."
)


def shap_waterfall_for_era(era_slug, input_row):
    model, scaler, explainer = load_era(era_slug)
    X_scaled = scaler.transform(input_row)
    shap_values = explainer(X_scaled)
    shap_values.feature_names = [FEATURE_LABELS[f].split(" (")[0] for f in FEATURES]
    return shap_values


col_shap_a, col_shap_b = st.columns(2)
with col_shap_a:
    st.markdown(f"**{ERA_LABELS[era_a]}**")
    sv_a = shap_waterfall_for_era(era_a, input_row)
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(sv_a[0], show=False)
    st.pyplot(fig_a, clear_figure=True)
with col_shap_b:
    st.markdown(f"**{ERA_LABELS[era_b]}**")
    sv_b = shap_waterfall_for_era(era_b, input_row)
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(sv_b[0], show=False)
    st.pyplot(fig_b, clear_figure=True)

with st.expander("See the exact input profile used"):
    display_row = input_row.copy()
    display_row["revenue_eur"] = display_row["revenue_eur"] / 1_000_000
    display_row["annual_wage_eur"] = display_row["annual_wage_eur"] / 1_000_000
    display_row["ucl_encoded"] = display_row["ucl_encoded"].apply(lambda i: UCL_STAGES[int(i)])
    st.dataframe(display_row.rename(columns=FEATURE_LABELS), hide_index=True)