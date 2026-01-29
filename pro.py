import streamlit as st
import pandas as pd
import joblib

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Football Player Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

# ------------------ Load Model ------------------
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ------------------ App Header ------------------
st.title("⚽ Football Player Market Value Predictor")
st.markdown(
    "Estimate a football player's **market value (€ Millions)** using performance, position, and league data."
)
st.divider()

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("📋 Player Information")

with st.sidebar.form("player_form"):

    st.subheader("Basic Info")
    age = st.number_input("Age", min_value=15, max_value=45, step=1)
    position = st.selectbox("Position", ["GK", "DEF", "MID", "FWD"])
    league = st.selectbox("League", ["EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"])

    st.subheader("Team & Contract")
    team_rank = st.number_input("Team Rank", min_value=1)
    contract_years_left = st.number_input("Contract Years Left", min_value=0)

    st.subheader("Match Stats")
    matches_played = st.number_input("Matches Played", min_value=0)
    minutes_played = st.number_input("Minutes Played", min_value=0)

    st.subheader("Performance Stats")
    goals = st.number_input("Goals", min_value=0)
    assists = st.number_input("Assists", min_value=0)
    saves = st.number_input("Saves", min_value=0)
    clean_sheets = st.number_input("Clean Sheets", min_value=0)
    tackles = st.number_input("Tackles", min_value=0)
    interceptions = st.number_input("Interceptions", min_value=0)

    submitted = st.form_submit_button("🚀 Predict Market Value")

# ------------------ Prediction Logic ------------------
if submitted:

    input_values = {
        "age": age,
        "matches_played": matches_played,
        "minutes_played": minutes_played,
        "team_rank": team_rank,
        "contract_years_left": contract_years_left,
        "goals": goals,
        "assists": assists,
        "saves": saves,
        "clean_sheets": clean_sheets,
        "tackles": tackles,
        "interceptions": interceptions
    }

    final_df = pd.DataFrame(0, index=[0], columns=model_columns)

    for col, val in input_values.items():
        if col in final_df.columns:
            final_df.loc[0, col] = val

    pos_col = f"position_{position}"
    league_col = f"league_{league}"

    if pos_col in final_df.columns:
        final_df.loc[0, pos_col] = 1

    if league_col in final_df.columns:
        final_df.loc[0, league_col] = 1

    final_df = final_df.astype(float)

    prediction = model.predict(final_df)[0]

    # ------------------ Output ------------------
    st.success("✅ Prediction Complete")

    st.metric(
        label="💰 Estimated Market Value",
        value=f"€ {prediction:,.2f} Million"
    )

    with st.expander("🔍 View Input Feature Vector"):
        st.dataframe(final_df)
