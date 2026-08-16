# The Production Function Time Machine
Interactive Streamlit dashboard built as the artefact for the MSc Data Science dissertation *"The Shifting Production Function of Sporting Success: A Longitudinal Machine Learning Analysis of European Football, 2010–2025."*
## Live Demo

[Open the Production Function Time Machine](https://football-time-machine.streamlit.app/)

The dashboard is deployed as a Streamlit application and can be used directly in the browser without local installation.
## What it does
The user enters a club's financial profile (revenue, wage bill, squad value, net transfer spend, squad age, Champions League depth) — either manually or by loading a real club-season from the study's 28-club panel — and compares the model's predicted standing, and the SHAP-based reasoning behind it, across two of the study's regulatory eras (Pre-FFP, Post-FFP, Post-COVID). The comparison demonstrates the dissertation's central finding: the relationship between financial resources and sporting success is not stable over time.
## How the underlying models were built
Three separate Random Forest models were trained in advance, one per regulatory era, using the same six input variables and hyperparameters as the main rolling-window analysis in the dissertation (Section 3.4). Each era's model, feature scaler, and SHAP explainer are saved together (`joblib`) so preprocessing at prediction time exactly matches training. Training code lives in the dissertation's analysis notebooks (not included in this repository); this repo contains only the dashboard and the pre-trained model artefacts needed to run it.
## Repository structure
master_panel_clean.csv - cleaned panel dataset (28 clubs, 2010-2025)
requirements.txt - Python dependencies
dashboard/
app.py - Streamlit application
models/ - pre-trained model, scaler and SHAP explainer per era
## Running it locally
pip install -r requirements.txt
streamlit run dashboard/app.py
## Author
Chloi Chatzinota — MSc Data Science, Coventry University (7005SCN Individual Research Project)
