import joblib
# After training the rf_model...
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(y_test, 'y_test.joblib')
joblib.dump(rf_model.predict_proba(X_test), 'probabilities.joblib')

import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd

# Attempt to load the trained model and related data
try:
    rf_model = joblib.load('rf_model.joblib')
except FileNotFoundError:
    model_path = os.path.join('C:/Users/Channing/PycharmProjects/pythonProject', 'rf_model.joblib')
    rf_model = joblib.load(model_path)

try:
    y_test = joblib.load('y_test.joblib')
except FileNotFoundError:
    y_test_path = os.path.join('C:/Users/Channing/PycharmProjects/pythonProject', 'y_test.joblib')
    y_test = joblib.load(y_test_path)

try:
    probabilities = joblib.load('probabilities.joblib')
except FileNotFoundError:
    probabilities_path = os.path.join('C:/Users/Channing/PycharmProjects/pythonProject', 'probabilities.joblib')
    probabilities = joblib.load(probabilities_path)



# Define the function that uses your model to make predictions
def predict_churn(input_data):
    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)
    return prediction, probability

# Streamlit page title
st.title('Customer Churn Prediction App')

# Displaying feature importances
if st.checkbox('Show Feature Importances'):
    feature_importances = pd.Series(rf_model.feature_importances_, index=[
        'Current Equipment Days', 'Monthly Minutes', 'Monthly Revenue', 'Months in Service',
        'Unanswered Calls', 'Outbound Calls', 'Inbound Calls', 'Age'
    ])
    fig, ax = plt.subplots()
    feature_importances.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title('Feature Importances in Predicting Churn')
    st.pyplot(fig)

# Input fields for features
input_features = []
feature_names = ['Current Equipment Days', 'Monthly Minutes', 'Monthly Revenue', 'Months in Service',
                 'Unanswered Calls', 'Outbound Calls', 'Inbound Calls', 'Age']
for feature in feature_names:
    input_features.append(st.number_input(f'Enter {feature}', value=0))

# Button to make prediction
if st.button('Predict Churn'):
    input_data = np.array([input_features])
    prediction, probability = predict_churn(input_data)

    # Display prediction and probability
    churn_status = 'likely to churn' if prediction[0] == 1 else 'unlikely to churn'
    st.success(f'This customer is {churn_status} with a probability of {probability[0][prediction[0]]:.2f}.')

    # Plotting the probability bar chart
    fig, ax = plt.subplots()
    ax.bar(['Stay', 'Churn'], probability[0], color=['green', 'red'])
    plt.ylabel('Probability')
    plt.title('Probability of Customer Churn')
    st.pyplot(fig)


