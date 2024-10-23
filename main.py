import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

# Initialize OpenAI client with the Groq API key from environment variables
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get('GROQ_API_KEY')
)

# Function to load pre-trained models from pickle files
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load all machine learning models
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_feature_engineered_model = load_model('xgboost_mode-featureEngineered.pkl')

# Prepare customer input data for model prediction
def prepare_input(credit_score, location, gender, age, tenure, balance, 
                  num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# Generate model predictions and display them using Streamlit
def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    # Display gauge chart with average probability
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    # Display individual model probabilities in a bar chart
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability

# Generate a detailed explanation of the prediction using OpenAI
def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in 
    interpreting and explaining predictions of machine learning models.

    A customer named {surname} has a {round(probability * 100, 1)}% probability of churning, 
    based on the information provided below:

    {input_dict}

    Feature importance for predicting churn:
    NumOfProducts           | 0.323888
    IsActiveMember          | 0.164146
    Age                     | 0.109550
    Geography_Germany       | 0.091373
    Balance                 | 0.052786
    Geography_France        | 0.046463
    Gender_Female           | 0.045283
    Geography_Spain         | 0.036855
    CreditScore             | 0.035005
    EstimatedSalary         | 0.032655

    If the customer has over a 40% risk of churning, explain why they are at risk.
    Otherwise, explain why they are not at risk of churning.
    """

    print("EXPLANATION PROMPT", prompt)

    response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Generate a personalized email to retain the customer using OpenAI
def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. A customer named {surname} has a 
    {round(probability * 100, 1)}% probability of churning.

    Customer Information:
    {input_dict}

    Explanation:
    {explanation}

    Write a personalized email offering incentives to retain this customer. Use bullet points
    for the incentives. Avoid mentioning the churn probability or the machine learning model.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nEMAIL PROMPT", prompt)
    return response.choices[0].message.content

# Streamlit application setup
st.title("Customer Churn Prediction")

# Load customer data
df = pd.read_csv("churn.csv")

# Display a dropdown to select a customer
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    # Extract selected customer details
    selected_customer_id = int(selected_customer_option.split("-")[0])
    selected_surname = selected_customer_option.split("-")[1].strip()
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    # Display customer details and input fields
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, int(selected_customer['CreditScore']))
        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"].index(selected_customer['Geography']))
        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == 'Male' else 1)
        age = st.number_input("Age", 18, 100, int(selected_customer['Age']))
        tenure = st.number_input("Tenure (years)", 0, 50, int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance", 0.0, value=float(selected_customer['Balance']))
        num_products = st.number_input("Number of Products", 1, 10, int(selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card", bool(selected_customer['HasCrCard']))
        is_active_member = st.checkbox("Is Active Member", bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input("Estimated Salary", 0.0, value=float(selected_customer['EstimatedSalary']))

    # Prepare input data and make predictions
    input_df, input_dict = prepare_input(
        credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary
    )

    avg_probability = make_predictions(input_df, input_dict)

    # Display prediction explanation
    explanation = explain_prediction(avg_probability, input_dict, selected_surname)
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    # Generate and display personalized email
    email = generate_email(avg_probability, input_dict, explanation, selected_surname)
    st.subheader("Personalized Email")
    st.markdown(email)
