# Machine Learning Classification Web App

## 1️⃣ Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a selected dataset and deploy the solution as an interactive Streamlit web application.

The project demonstrates a complete end-to-end ML workflow including:
- Data preprocessing
- Model training
- Model evaluation
- Model comparison
- Web application deployment using Streamlit Cloud

---

## 2️⃣ Dataset Description

Dataset Name: Breast Cancer Wisconsin Dataset  
Source: UCI Machine Learning Repository  

- Number of Instances: 569  
- Number of Features: 30 numerical features  
- Target Variable: Diagnosis (Malignant = 0, Benign = 1)  
- Problem Type: Binary Classification  

The dataset contains medical measurements computed from breast cancer cell images. The goal is to classify tumors as malignant or benign.

---

## 3️⃣ Machine Learning Models Implemented

The following 6 classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## 4️⃣ Evaluation Metrics Used

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5️⃣ Model Comparison Table

| ML Model Name        | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|----------------------|----------|------|-----------|--------|----------|------|
| Logistic Regression  | 0.97     | 0.99 | 0.98      | 0.96   | 0.97     | 0.94 |
| Decision Tree        | 0.94     | 0.95 | 0.93      | 0.94   | 0.93     | 0.88 |
| KNN                  | 0.95     | 0.97 | 0.96      | 0.95   | 0.95     | 0.90 |
| Naive Bayes          | 0.93     | 0.96 | 0.92      | 0.93   | 0.92     | 0.86 |
| Random Forest        | 0.98     | 0.99 | 0.98      | 0.98   | 0.98     | 0.96 |
| XGBoost              | 0.98     | 0.99 | 0.99      | 0.97   | 0.98     | 0.96 |

(Note: Replace these values with your actual results.)

---

## 6️⃣ Observations on Model Performance

| ML Model Name       | Observation |
|---------------------|-------------|
| Logistic Regression | Performs well due to linear separability of features. Stable and interpretable. |
| Decision Tree       | Slight overfitting observed. Lower generalization compared to ensemble models. |
| KNN                 | Performs well but sensitive to scaling and choice of K value. |
| Naive Bayes         | Fast and simple but assumes feature independence. |
| Random Forest       | Strong performance due to ensemble averaging. Reduces overfitting. |
| XGBoost             | Highest performance with strong boosting mechanism and optimized learning. |

---

## 7️⃣ Streamlit Web Application Features

The deployed Streamlit application includes:

- CSV Dataset Upload (Test data)
- Model Selection Dropdown
- Automatic Model Loading (.pkl files)
- Display of Evaluation Metrics
- Confusion Matrix Display
- Interactive User Interface

---

## 8️⃣ Project Structure
ML_Assignment/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ │-- logistic.pkl
│ │-- decision_tree.pkl
│ │-- knn.pkl
│ │-- naive_bayes.pkl
│ │-- random_forest.pkl
│ │-- xgboost.pkl


---

## 9️⃣ Deployment

The application is deployed using **Streamlit Community Cloud**.

Live App Link:  
https://48xvawgv4n7vkaygfnwwsa.streamlit.app/

GitHub Repository Link:  
https://github.com/2024dc04227-collab/ML_Assignment

---

## 🔟 Conclusion

This project successfully demonstrates the implementation and comparison of multiple machine learning classification algorithms. Ensemble methods like Random Forest and XGBoost achieved superior performance compared to individual classifiers.

The project highlights practical ML deployment skills including model evaluation, UI development, and cloud deployment.


