# 🧠 Mental Health Prediction Web App

This is a **Machine Learning powered Streamlit web app** that predicts an individual’s mental health condition based on survey inputs.  
The model (`mental_health_best_model.joblib`) is trained on relevant features to analyze patterns and provide predictions in real-time.

---

## 🚀 Features
- 🏥 Predicts likelihood of mental health condition based on user responses.  
- 🎛️ Simple and interactive **Streamlit UI**.  
- 📦 **Pre-trained ML model** loaded with Joblib.  
- 💡 Useful for **awareness and educational purposes**.  

---

## 🛠️ Tech Stack
- Python 3.12+  
- Streamlit (Web framework)  
- Scikit-learn (Machine Learning)  
- Joblib (Model serialization)  
- Pandas / NumPy (Data handling)  

---

## 📂 Project Structure
Mental-Health-Prediction/
│── app.py # Streamlit app
│── mental_health_best_model.joblib # Pre-trained ML model
│── requirements.txt # Required Python libraries
│── README.md # Project documentation


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/Mental-Health-Prediction.git
cd Mental-Health-Prediction
```

### 2️⃣ Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate     # For Mac/Linux
venv\Scripts\activate        # For Windows
```

### 3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
Then open the given localhost URL in your browser (e.g., http://localhost:8501).

### 📌 Example Inputs & Outputs

Inputs: Age, work environment, family history, previous mental health conditions, etc.

Output: Probability or classification of mental health condition.

### ⚠️ Disclaimer

This project is for educational and awareness purposes only and not a medical diagnostic tool.
Please consult a licensed healthcare professional for mental health concerns.


