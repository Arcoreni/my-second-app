import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---- Generate Dataset ----
@st.cache_data
def generate_dataset():
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    columns = [f"Feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["Target"] = y
    return df

df = generate_dataset()

# ---- Prepare Training Data ----
X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train Model ----
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---- Streamlit App UI ----
st.title("🧠 Interactive ML Classifier App")
st.markdown("Adjust the sliders to input feature values and see the model prediction in real-time.")

# ---- Sidebar Input ----
st.sidebar.header("Input Features")

def user_input_features():
    input_data = {}
    for column in X.columns:
        col_min = float(df[column].min())
        col_max = float(df[column].max())
        col_mean = float(df[column].mean())
        input_data[column] = st.sidebar.slider(column, col_min, col_max, col_mean)
    return pd.DataFrame([input_data])

input_df = user_input_features()

# ---- Make Prediction ----
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# ---- Display Results ----
st.subheader("Prediction")
st.write(f"**Predicted Class:** {prediction}")
st.write("**Prediction Probability:**")
st.write({f"Class {i}": f"{prob*100:.2f}%" for i, prob in enumerate(prediction_proba)})

# ---- Dataset Preview ----
with st.expander("📊 Preview Synthetic Dataset"):
    st.dataframe(df.head())

# ---- Model Performance ----
st.subheader("Model Accuracy")
accuracy = model.score(X_test, y_test)
st.write(f"**Accuracy on Test Set:** {accuracy:.2%}")