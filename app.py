import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

MODEL_PATH = "model/titanic_model.joblib"
META_PATH = "model/metadata.json"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)
    return model, metadata


st.title("🚢 Titanic Survival Predictor")
st.write(
    "This app uses a trained machine learning pipeline (preprocessing + "
    "classifier saved together with `joblib`) to predict whether a "
    "passenger would have survived the Titanic disaster."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run `python train_model.py` first to train "
        "and save the model, then relaunch this app."
    )
    st.stop()

model, metadata = load_model()

if metadata:
    with st.expander("Model info"):
        st.write(f"**Best model:** {metadata.get('best_model')}")
        st.json(metadata.get("metrics", {}))

st.subheader("Enter passenger details")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                           help="1 = First, 2 = Second, 3 = Third")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 90, 29)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")
with col2:
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children aboard", 0, 10, 0)
    fare = st.number_input("Fare paid", 0.0, 600.0, 32.0, step=1.0)

family_size = sibsp + parch + 1

input_df = pd.DataFrame([{
    "pclass": pclass,
    "sex": sex,
    "age": age,
    "sibsp": sibsp,
    "parch": parch,
    "fare": fare,
    "embarked": embarked,
    "family_size": family_size,
}])

if st.button("Predict", type="primary"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"✅ Predicted: **Survived** (probability: {proba:.1%})")
    else:
        st.error(f"❌ Predicted: **Did not survive** (probability of survival: {proba:.1%})")

    st.progress(min(max(proba, 0.0), 1.0))
    with st.expander("Input passed to the model"):
        st.dataframe(input_df)
