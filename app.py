import streamlit as st
import joblib
import re
import string
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Mental Health Prediction", layout="centered")

# -----------------------
# Define the preprocessing function used in training
# -----------------------
def simple_clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# -----------------------
# Softmax function
# -----------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("mental_health_balanced_model.joblib")

model = load_model()

# -----------------------
# Define colors for classes
# -----------------------
class_colors = {
    "Normal": "#2ecc71",  # green
    "Depression": "#e74c3c",  # red
    "Anxiety": "#e67e22",  # orange
    "Stress": "#9b59b6",  # purple
    "Bipolar": "#3498db",  # blue
    "Suicidal": "#8e2c2c",  # dark red/brown
    "Personality disorder": "#f78fb3"  # pink
}

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 Mental Health Prediction App")
st.write("Enter your thoughts below and the model will predict the hidden emotion.")

user_input = st.text_area("Your thoughts:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        cleaned_input = simple_clean(user_input)
        prediction = model.predict([cleaned_input])[0]

        if hasattr(model, "decision_function"):
            scores = model.decision_function([cleaned_input])[0]
            probs = softmax(scores) * 100
            class_labels = model.classes_

            sorted_indices = np.argsort(probs)[::-1]
            output_lines = []
            pie_labels = []
            pie_probs = []
            pie_colors = []

            for i in sorted_indices:
                label = class_labels[i]
                prob = probs[i]
                output_lines.append(f"{label} - {prob:.1f}%")
                pie_labels.append(f"{label} ({prob:.1f}%)")
                pie_probs.append(prob)
                pie_colors.append(class_colors.get(label, "gray"))

            # Add overall
            output_lines.append(f"Overall - {prediction}")

            # Show predictions with colored box
            color_hex = class_colors.get(prediction, "#34495e")
            predictions_text = "<br>".join(output_lines)
            st.markdown(
                f"""
                <div style="background-color:{color_hex}; padding:15px; border-radius:10px; color:white; font-size:16px;">
                    {predictions_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            # -----------------------
            # Fancy Pie Chart
            # -----------------------
            fig, ax = plt.subplots(figsize=(6, 6))
            explode = [0.1 if i == sorted_indices[0] else 0 for i in range(len(pie_probs))]  # highlight top class

            wedges, texts, autotexts = ax.pie(
                pie_probs,
                labels=pie_labels,
                autopct='%1.1f%%',
                startangle=140,
                colors=pie_colors,
                explode=explode,
                shadow=True,
                textprops={'fontsize': 10, 'color': "black"}
            )

            ax.axis("equal")
            plt.setp(autotexts, size=11, weight="bold", color="white")
            plt.setp(texts, size=11, weight="bold")
            st.pyplot(fig)

        else:
            st.success(f"### 🎯 Prediction: **{prediction}**")
