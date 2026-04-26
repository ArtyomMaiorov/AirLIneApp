import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# page config
st.set_page_config(
    page_title="Flight Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
)

# load cached artefacts
@st.cache_resource
def load_artefacts():
    preprocessor = joblib.load("deployment/preprocessor.joblib")
    model = joblib.load("deployment/xgb_model.joblib")
    with open("deployment/artefacts.json") as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(model)
    return preprocessor, model, explainer, meta

preprocessor, model, explainer, meta = load_artefacts()

NUMERIC_FEATURES = meta["numeric_features"]
CATEGORICAL_FEATURES = meta["categorical_features"]
ALL_FEATURE_NAMES = meta["all_feature_names"]

# header
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown(
    "Enter a passenger's details below. The model will predict whether they are "
    "**Satisfied** or **Neutral / Dissatisfied** and explain the key drivers using SHAP."
)
st.divider()

# input form
st.subheader("Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics & Trip**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
    age = st.slider("Age", 7, 85, 35)
    flight_distance = st.slider("Flight Distance (miles)", 50, 5000, 1000)

with col2:
    st.markdown("**Service Ratings** (0-5 scale)")
    wifi = st.slider("Inflight WiFi Service",              0, 5, 3)
    time_conven = st.slider("Departure/Arrival Time Convenient",  0, 5, 3)
    online_book = st.slider("Ease of Online Booking",             0, 5, 3)
    gate_loc = st.slider("Gate Location",                      0, 5, 3)
    food_drink = st.slider("Food and Drink",                     0, 5, 3)
    online_board = st.slider("Online Boarding",                    0, 5, 3)
    seat_comfort = st.slider("Seat Comfort",                       0, 5, 3)

with col3:
    st.markdown("**More Service Ratings & Delays**")
    entertainment = st.slider("Inflight Entertainment",  0, 5, 3)
    onboard_svc = st.slider("On-board Service",        0, 5, 3)
    leg_room = st.slider("Leg Room Service",        0, 5, 3)
    baggage = st.slider("Baggage Handling",        0, 5, 3)
    checkin = st.slider("Check-in Service",        0, 5, 3)
    inflight_svc = st.slider("Inflight Service",        0, 5, 3)
    cleanliness = st.slider("Cleanliness",             0, 5, 3)
    dep_delay = st.number_input("Departure Delay (minutes)", 0, 1600, 0)
    arr_delay = st.number_input("Arrival Delay (minutes)",   0, 1600, 0)

st.divider()

# predict button
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict_btn = st.button("🔍 Predict Satisfaction", type="primary")

if predict_btn:
    input_dict = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": time_conven,
        "Ease of Online booking": online_book,
        "Gate location": gate_loc,
        "Food and drink": food_drink,
        "Online boarding": online_board,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": entertainment,
        "On-board service": onboard_svc,
        "Leg room service": leg_room,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Inflight service": inflight_svc,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay,
    }
    input_df = pd.DataFrame([input_dict])
    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0]

    st.session_state["prediction"] = int(prediction)
    st.session_state["probability"] = probability.tolist()
    st.session_state["X_processed"] = X_processed.tolist()

# display results
if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    probability = st.session_state["probability"]
    X_processed = np.array(st.session_state["X_processed"])

    satisfied_prob = probability[1]
    not_satisfied_prob = probability[0]

    # result banner
    st.subheader("Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.success("✅ Satisfied")
        else:
            st.error("❌ Neutral / Dissatisfied")

        st.metric("Satisfied probability", f"{satisfied_prob:.1%}")
        st.metric("Neutral/Dissatisfied probability", f"{not_satisfied_prob:.1%}")

    with res_col2:
        fig_bar, ax = plt.subplots(figsize=(5, 1.2))
        ax.barh([""], [satisfied_prob], color="#2ecc71", label="Satisfied")
        ax.barh([""], [not_satisfied_prob], left=[satisfied_prob],
                color="#e74c3c", label="Neutral/Dissatisfied")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.axvline(0.5, color="white", linewidth=1.5, linestyle="--")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Predicted probability", fontsize=10)
        fig_bar.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    st.divider()

    # SHAP explanation
    st.subheader("Why did the model predict this?")
    st.markdown(
        "The waterfall plot below shows each feature's contribution to this prediction. "
        "**Red bars** push the prediction toward *Satisfied*; "
        "**blue bars** push toward *Neutral/Dissatisfied*. "
        "The features are ranked by absolute impact."
    )

    X_shap = pd.DataFrame(X_processed, columns=ALL_FEATURE_NAMES)
    shap_values = explainer(X_shap)

    shap_col, _ = st.columns([1, 1])
    with shap_col:
        fig_shap, ax_shap = plt.subplots(figsize=(7, 4))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.title("SHAP Waterfall — Feature contributions to this prediction", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

    # plain-english summary
    st.divider()
    st.subheader("Plain-English Summary")

    shap_vals = shap_values[0].values
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    top3_pos = [(ALL_FEATURE_NAMES[i], shap_vals[i]) for i in sorted_idx if shap_vals[i] > 0][:3]
    top3_neg = [(ALL_FEATURE_NAMES[i], shap_vals[i]) for i in sorted_idx if shap_vals[i] < 0][:3]

    verdict = "satisfied" if prediction == 1 else "neutral or dissatisfied"
    st.markdown(
        f"The model predicts this passenger is **{verdict}** with **{max(satisfied_prob, not_satisfied_prob):.1%} confidence**.")

    if top3_pos:
        drivers = ", ".join([f"**{n}**" for n, _ in top3_pos])
        st.markdown(f"The strongest factors pushing toward *Satisfied* were: {drivers}.")
    if top3_neg:
        drivers = ", ".join([f"**{n}**" for n, _ in top3_neg])
        st.markdown(f"The strongest factors pushing toward *Neutral/Dissatisfied* were: {drivers}.")

# footer
st.divider()
st.caption(
    "Flight Analytics - MSc Machine Learning Group Project. Created by Dina Kaziyeva, Artyom Maiorov, Arlen Maker. "
    "Model: XGBoost (tuned, early stopping). "
    "Dataset: Airline Passenger Satisfaction from Kaggle."
)
