import streamlit as st
import joblib
import re
import string
import numpy as np

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Mental Health Prediction", layout="centered")

# -----------------------
# Define the preprocessing function used in training
# -----------------------
def simple_clean(text):
    """
    Custom preprocessing function used during training (Balanced.ipynb).
    Must exactly match what was applied before TF-IDF in training.
    """
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove links
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.strip()
    return text

# -----------------------
# Softmax function (convert decision scores -> probabilities)
# -----------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# -----------------------
# Load trained model (Balanced version)
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("mental_health_balanced_model.joblib")

model = load_model()

# -----------------------
# Streamlit UI Elements
# -----------------------
st.title("🧠 Mental Health Prediction App")
st.write("Enter your thoughts below and the model will predict the hidden emotion.")

# Input text box
user_input = st.text_area("Your thoughts:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Apply preprocessing
        cleaned_input = simple_clean(user_input)

        # Predict label
        prediction = model.predict([cleaned_input])[0]

        # Get decision scores & convert to probabilities
        if hasattr(model, "decision_function"):
            scores = model.decision_function([cleaned_input])[0]
            probs = softmax(scores) * 100
            class_labels = model.classes_

            # Sort classes by probability
            sorted_indices = np.argsort(probs)[::-1]
            output_lines = []
            for i in sorted_indices:
                output_lines.append(f"{class_labels[i]} - {probs[i]:.1f}%")

            # Add overall prediction
            output_lines.append(f"Overall - {prediction}")

            # Join lines with newlines
            predictions_text = "\n".join(output_lines)

            # Display neatly in vertical format
            st.success(predictions_text)
        else:
            st.success(f"### 🎯 Prediction: **{prediction}**")
