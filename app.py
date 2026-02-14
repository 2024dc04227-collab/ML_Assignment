import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score

st.title("ML Classification App")

st.write("Upload dataset and select model")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression","Decision Tree","KNN","Naive Bayes","Random Forest","XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.write(data.head())

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    if model_choice=="Logistic Regression":
        model = pickle.load(open("model/logistic.pkl","rb"))
    elif model_choice=="Decision Tree":
        model = pickle.load(open("model/tree.pkl","rb"))
    elif model_choice=="KNN":
        model = pickle.load(open("model/knn.pkl","rb"))
    elif model_choice=="Naive Bayes":
        model = pickle.load(open("model/nb.pkl","rb"))
    elif model_choice=="Random Forest":
        model = pickle.load(open("model/rf.pkl","rb"))
    else:
        model = pickle.load(open("model/xgb.pkl","rb"))

    pred = model.predict(X)

    st.subheader("Results")

    st.write("Accuracy:", accuracy_score(y,pred))
    st.write("Precision:", precision_score(y,pred))
    st.write("Recall:", recall_score(y,pred))
    st.write("F1 Score:", f1_score(y,pred))
    st.write("MCC:", matthews_corrcoef(y,pred))
    st.write("AUC:", roc_auc_score(y,pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y,pred))
