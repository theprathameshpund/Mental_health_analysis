import streamlit as st
import joblib
import re
import string

# -----------------------
# Define the preprocessing function used in training
# -----------------------
def simple_clean(text):
    """
    Custom preprocessing function used during training.
    Make sure this matches exactly what you used earlier.
    """
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove links
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.strip()
    return text

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("mental_health_best_model.joblib")

model = load_model()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Mental Health Prediction", layout="centered")

st.title("🧠 Mental Health Prediction App")
st.write("Enter your thoughts below and the model will try to predict the hidden emotion.")

# Input text box
user_input = st.text_area("Your thoughts:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Preprocess and predict
        prediction = model.predict([user_input])[0]
        st.success(f"### 🎯 Prediction: **{prediction}**")
