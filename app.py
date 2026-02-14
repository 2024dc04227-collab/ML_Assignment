import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
import os
import glob

st.title("ML Classification App")
st.write("Upload dataset and select model")

# Show current working directory (safe for deployment)
try:
    cwd = os.getcwd()
    st.write("Current working directory:", cwd)
except Exception as e:
    st.warning(f"Cannot access working directory: {e}")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Dynamically find all model files in 'model/' folder
model_folder = "model"
if not os.path.exists(model_folder):
    st.error(f"Model folder not found: {model_folder}")
    st.stop()

model_files = glob.glob(os.path.join(model_folder, "*.pkl"))

if not model_files:
    st.error("No model files (.pkl) found in the model folder!")
    st.stop()

# Create a mapping of friendly names to file paths
model_map = {os.path.splitext(os.path.basename(f))[0]: f for f in model_files}

# Dropdown to select model
model_choice = st.selectbox("Choose Model", list(model_map.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Load selected model safely
    model_path = model_map[model_choice]
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model {model_choice}: {e}")
        st.stop()

    # Make predictions
    pred = model.predict(X)

    # Display metrics
    st.subheader("Results")
    st.write("Accuracy:", accuracy_score(y, pred))
    st.write("Precision:", precision_score(y, pred))
    st.write("Recall:", recall_score(y, pred))
    st.write("F1 Score:", f1_score(y, pred))
    st.write("MCC:", matthews_corrcoef(y, pred))
    st.write("AUC:", roc_auc_score(y, pred))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, pred))
