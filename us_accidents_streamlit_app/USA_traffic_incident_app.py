import warnings
warnings.filterwarnings("ignore")  # optional: suppress version and shap warnings

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import pydeck as pdk
from pathlib import Path


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="US Accidents – Severity Prediction",
    layout="wide",
)

st.title("🚦 US Accidents – Severity Prediciton Dashboard")
st.caption("Binary model: Severity_Bin = 0 (non-severe), 1 = severe")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745 !important;  /* green */
        color: white !important;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #218838 !important;  /* darker green on hover */
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Map from state codes used in the model to full names in the GeoJSON
STATE_FULL_NAME = {
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

@st.cache_resource
def load_state_geojson():
    base_dir = Path(__file__).parent
    geo_path = base_dir / "us_states.geojson"
    with open(geo_path, "r") as f:
        gj = json.load(f)
    return gj

# ---------------- LOAD MODEL & META ----------------
@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).parent

    model_path = base_dir / "us_accidents_logreg_sklearn.pkl"
    meta_path = base_dir / "us_accidents_logreg_meta.pkl"
    
    sk_model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return sk_model, meta

sk_model, meta = load_artifacts()

numeric_cols = meta["numeric_cols"]
categorical_cols = meta["categorical_cols"]
scaler_model = meta["scaler_model"]
feature_names = meta["feature_names"]

CLASS_LABELS = list(sk_model.classes_)
POS_CLASS = 1
POS_INDEX = CLASS_LABELS.index(POS_CLASS)

# ---------------- SHAP EXPLAINER ----------------
@st.cache_resource
def load_explainer():
    # very simple background: all features = 0 (already scaled numerics)
    background = np.zeros((1, len(feature_names)))
    explainer = shap.LinearExplainer(
        sk_model,
        background,
        feature_names=feature_names
    )
    return explainer

explainer = load_explainer()

# ---------------- PREPROCESSING ----------------
def preprocess_single(raw_row: dict) -> np.ndarray:
    """
    raw_row: dict with raw values for numeric + categorical features.
    Numeric features are scaled using scaler_model.
    Categorical features are taken as provided (0/1 or numeric).
    Any missing feature defaults to 0.
    """
    # numeric
    scaled_numeric = []
    for c in numeric_cols:
        val = float(raw_row.get(c, 0.0))
        mean, std = scaler_model[c]
        scaled_numeric.append((val - mean) / std)

    # categorical
    cat_vals = [raw_row.get(c, 0.0) for c in categorical_cols]

    x = np.array(scaled_numeric + cat_vals).reshape(1, -1)
    return x

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("ℹ️ About this app")
# st.sidebar.markdown("---")
# st.sidebar.write("**Model classes:**", CLASS_LABELS)
# st.sidebar.write("**# features:**", len(feature_names))

st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider(
    "Decision threshold for predicting severe (class 1)",
    0.0, 1.0, 0.5, 0.01,
    help="Prediction is classified as severe if P(severe) ≥ threshold."
)

st.sidebar.markdown(
    """
    _Adjust the threshold depending on whether you want the model  
    to be more sensitive (recall-focused) or more precise._
    """
)

# ---------------- MAIN UI: SINGLE SCENARIO ----------------
st.subheader("🔍 Single accident scenario")

st.markdown(
    """
Define a single accident scenario below.  
The model will predict the probability of a **severe accident (1)**,  
highlight the selected **state**, and show **SHAP feature contributions**.
"""
)

# ---- 1) State and weather FIRST ----
row1_col1, row1_col2 = st.columns(2)

# State
state_options = sorted(
    [s.replace("State_", "") for s in categorical_cols if s.startswith("State_")]
)
default_state_idx = state_options.index("GA") if "GA" in state_options else 0

with row1_col1:
    selected_state = st.selectbox(
        "State",
        options=state_options,
        index=default_state_idx,
    )

# Weather
weather_options = sorted(
    [w.replace("Weather_Category_", "") for w in categorical_cols if w.startswith("Weather_Category_")]
)
default_weather_idx = weather_options.index("Clear") if "Clear" in weather_options else 0

with row1_col2:
    selected_weather = st.selectbox(
        "Weather category",
        options=weather_options,
        index=default_weather_idx,
    )

# ---- 2) Time features (month, day-of-week, hour) ----
row2_col1, row2_col2, row2_col3 = st.columns(3)

month_options = [str(m) for m in range(1, 13)]
dow_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hour_options = list(range(0, 24))

with row2_col1:
    selected_month = st.selectbox("Month (1–12)", options=month_options, index=0)

with row2_col2:
    selected_dow = st.selectbox("Day of week", options=dow_options, index=0)

with row2_col3:
    selected_hour = st.selectbox("Hour of day (0–23)", options=hour_options, index=8)

# ---- 3) Numeric weather/visibility features BELOW ----
col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input("Temperature(F)", value=60.0, step=1.0)
    humidity = st.number_input("Humidity(%)", value=65.0, step=1.0)

with col2:
    pressure = st.number_input("Pressure(in)", value=29.6, step=0.1)
    visibility = st.number_input("Visibility(mi)", value=9.0, step=0.5)

with col3:
    log_wind = st.number_input("LogWindSpeed", value=2.0, step=0.1)

# ---- 5) Road environment booleans ----
st.markdown("**Road environment (True/False)**")
env_cols = [
    "Amenity_True",
    "Bump_True",
    "Crossing_True",
    "Give_Way_True",
    "Junction_True",
    "No_Exit_True",
    "Railway_True",
    "Roundabout_True",
    "Station_True",
    "Stop_True",
    "Traffic_Calming_True",
    "Traffic_Signal_True",
    "Daytime_True",
]

env_values = {}
env_col1, env_col2, env_col3 = st.columns(3)
for i, col in enumerate(env_cols):
    label = col.replace("_True", "")
    container = [env_col1, env_col2, env_col3][i % 3]
    default_val = (col == "Daytime_True")
    with container:
        env_values[col] = st.checkbox(label, value=default_val)

# ---------------- PREDICTION BUTTON ----------------
if st.button("Predict severity and explain"):
    # raw feature dict, everything else defaults to 0 in preprocess_single
    raw_row = {}

    # numeric
    raw_row["Temperature(F)"] = temp
    raw_row["Humidity(%)"] = humidity
    raw_row["Pressure(in)"] = pressure
    raw_row["Visibility(mi)"] = visibility
    raw_row["LogWindSpeed"] = log_wind

    # ---- City dummies: fixed to Other ----
    for c in categorical_cols:
        if c.startswith("City_"):
            raw_row[c] = 0.0
    if "City_Other" in categorical_cols:
        raw_row["City_Other"] = 1.0

    # ---- State one-hot ----
    for s in categorical_cols:
        if s.startswith("State_"):
            raw_row[s] = 0.0
    state_col = f"State_{selected_state}"
    if state_col in categorical_cols:
        raw_row[state_col] = 1.0

    # ---- Weather category one-hot ----
    for w in categorical_cols:
        if w.startswith("Weather_Category_"):
            raw_row[w] = 0.0
    weather_col = f"Weather_Category_{selected_weather}"
    if weather_col in categorical_cols:
        raw_row[weather_col] = 1.0

    # ---- Month ----
    for m in categorical_cols:
        if m.startswith("month_"):
            raw_row[m] = 0.0
    month_col = f"month_{selected_month}"
    if month_col in categorical_cols:
        raw_row[month_col] = 1.0

    # ---- Day of week ----
    for d in categorical_cols:
        if d.startswith("day_of_week_"):
            raw_row[d] = 0.0
    dow_map = {
        "Mon": "day_of_week_Mon",
        "Tue": "day_of_week_Tue",
        "Wed": "day_of_week_Wed",
        "Thu": "day_of_week_Thu",
        "Fri": "day_of_week_Fri",
        "Sat": "day_of_week_Sat",
        "Sun": "day_of_week_Sun",
    }
    dow_col = dow_map[selected_dow]
    if dow_col in categorical_cols:
        raw_row[dow_col] = 1.0

    # ---- Hour of day ----
    for h in categorical_cols:
        if h.startswith("hour_of_day_"):
            raw_row[h] = 0.0
    hour_col = f"hour_of_day_{selected_hour}"
    if hour_col in categorical_cols:
        raw_row[hour_col] = 1.0

    # ---- Environment booleans ----
    for col, val in env_values.items():
        raw_row[col] = 1.0 if val else 0.0

    # ---------------- PREDICT ----------------
    X = preprocess_single(raw_row)
    proba = sk_model.predict_proba(X)[0]
    p_severe = proba[POS_INDEX]
    pred_label = int(p_severe >= threshold)
    readable = "Severe (1)" if pred_label == 1 else "Non-severe (0)"

    # 🔥 create the 3 columns here (this guarantees map_col exists)
    pred_col, map_col, shap_col = st.columns([1.2, 1.3, 1.5])

    # --- Prediction summary ---
    with pred_col:
        st.markdown("### 🧮 Prediction")
        st.metric(
            label="Predicted class",
            value=f"{pred_label} — {readable}",
            delta=f"P(severe) = {p_severe:.3f}",
        )

        proba_df = pd.DataFrame(
            {"Class": CLASS_LABELS, "Probability": proba}
        ).set_index("Class")
        st.bar_chart(proba_df)

        st.caption(
            f"Decision rule: predict 1 (severe) if P(severe) ≥ {threshold:.2f}."
        )

    # --- Map: highlight entire selected state ---
    with map_col:
        st.markdown("### 🗺 Selected state (colored by prediction)")

        gj = load_state_geojson()

        full_name = STATE_FULL_NAME.get(selected_state)
        if full_name is None:
            st.error(f"Cannot map state code '{selected_state}' to full name.")
        else:
            # Find the GeoJSON feature for this state
            state_feature = None
            for feature in gj["features"]:
                if feature["properties"].get("name") == full_name:
                    state_feature = feature
                    break

            if state_feature is None:
                st.error(f"Could not find geometry for state '{full_name}' in GeoJSON.")
            else:
                # Center of selected state
                coords = np.array(state_feature["geometry"]["coordinates"][0])
                center_lon = coords[:, 0].mean()
                center_lat = coords[:, 1].mean()

                # 🔴🟢 choose color based on prediction
                if pred_label == 1:
                    # severe -> red
                    highlight_color = [220, 0, 0, 160]   # RGBA
                    label_text = "Predicted SEVERE (red)"
                else:
                    # non-severe -> green
                    highlight_color = [0, 160, 0, 160]
                    label_text = "Predicted NON-SEVERE (green)"

                # Layer 1: all states in light gray
                all_states_layer = pdk.Layer(
                    "GeoJsonLayer",
                    gj,
                    pickable=False,
                    stroked=True,
                    filled=True,
                    get_fill_color="[220, 220, 220, 120]",   # light gray
                    get_line_color="[150, 150, 150]",
                    line_width_min_pixels=1,
                )

                # Layer 2: selected state highlighted, color by prediction
                highlight_layer = pdk.Layer(
                    "GeoJsonLayer",
                    [state_feature],
                    pickable=True,
                    stroked=True,
                    filled=True,
                    get_fill_color=highlight_color,
                    get_line_color="[0, 90, 120]",
                    line_width_min_pixels=2,
                )

                view_state = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=5,
                    pitch=0,
                )

                
                deck = pdk.Deck(
                    layers=[all_states_layer, highlight_layer],
                    initial_view_state=view_state,
                    tooltip={"text": f"{full_name} — {label_text}"},
                    map_style="mapbox://styles/mapbox/light-v10",
                )

                st.pydeck_chart(deck)

        st.caption(
            f"State **{selected_state}**: red = model predicts severe (1), "
            "green = model predicts non-severe (0)."
        )

    # --- SHAP explanation ---
    with shap_col:
        st.markdown("### 📈 SHAP local explanation")

        shap_values = explainer(X)
        sv = shap_values.values[0]

        # Top K features by absolute SHAP
        K = 15
        idx = np.argsort(np.abs(sv))[::-1][:K]
        top_feats = np.array(feature_names)[idx]
        top_shap = sv[idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(top_feats[::-1], top_shap[::-1])
        ax.set_xlabel("SHAP value (impact on log-odds of severe=1)")
        ax.set_ylabel("Feature")
        ax.set_title("Top feature contributions for this scenario")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Positive SHAP values push towards **severe (1)**; "
            "negative values push towards **non-severe (0)**."
        )
