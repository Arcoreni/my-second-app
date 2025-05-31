import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# ⚠️ DISCLAIMER
# ---------------------------
st.title("🌈 GAG ML CLASSIFIER: 'Is Homosexual?' (Totally Fake 🃏)")
st.markdown("""
> ⚠️ **DISCLAIMER**: This app is a parody.  
> All data is randomly generated.  
> The model is meaningless and should not be used seriously for any purpose.

Enjoy responsibly.
""")

# ---------------------------
# 🎲 Generate Random Dataset
# ---------------------------
@st.cache_data
def generate_fake_data(n_samples=500):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 60, n_samples),
        "Education Level": np.random.randint(1, 6, n_samples),  # 1 to 5
        "Income Level": np.random.randint(2000, 12000, n_samples),
        "Hours Online per Day": np.random.uniform(0, 12, n_samples).round(1),
        "Satisfaction Score": np.random.randint(1, 11, n_samples),
        "Is Homosexual": np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    return df

df = generate_fake_data()

# ---------------------------
# 🧠 Train Model
# ---------------------------
X = df.drop("Is Homosexual", axis=1)
y = df["Is Homosexual"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# 🎛️ User Input via Sidebar
# ---------------------------
st.sidebar.header("Input Your (Totally Random) Info")

def user_input():
    age = st.sidebar.slider("Age", 18, 60, 30)
    edu = st.sidebar.slider("Education Level (1-Low to 5-High)", 1, 5, 3)
    income = st.sidebar.slider("Income Level", 2000, 12000, 5000)
    online = st.sidebar.slider("Hours Online per Day", 0.0, 12.0, 4.0)
    satisfaction = st.sidebar.slider("Satisfaction Score (1–10)", 1, 10, 5)
    return pd.DataFrame([{
        "Age": age,
        "Education Level": edu,
        "Income Level": income,
        "Hours Online per Day": online,
        "Satisfaction Score": satisfaction
    }])

input_df = user_input()

# ---------------------------
# 🔮 Make Prediction
# ---------------------------
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][prediction]

# ---------------------------
# 📢 Results
# ---------------------------
st.subheader("Prediction Result")
st.write(f"🔍 **Model Guess:** `{ 'Homosexual' if prediction == 1 else 'Not Homosexual' }`")
st.write(f"🤔 **Confidence:** `{proba:.2%}`")

with st.expander("📊 Peek at Fake Dataset"):
    st.dataframe(df.head())

st.subheader("Model Accuracy")
st.write(f"🎯 Test Accuracy: **{model.score(X_test, y_test):.2%}** (on completely random data 🤷)")
