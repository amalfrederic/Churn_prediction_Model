import streamlit as st
import joblib 
import pandas as pd
import numpy as np

l=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

model = joblib.load("../notebooks/model.pkl")
scaler = joblib.load("../notebooks/scaler.pkl")

st.title("Churn Prediction")
st.write("Welcome to my churn prediction model")

sc=st.checkbox("Senior Citizen")
tenure=st.slider("Tenure in Months: ",0,72,24)

monthly_charges=st.number_input("Monthly Charges",0.0,value=70.0)
total_charges=tenure*monthly_charges

gender=st.selectbox("Gender",["Select","Male","Female"])
st.write("Tick the Checkbox for yes")
partner=st.checkbox("Partner")
Dependents=st.checkbox("Dependants")
Phone_service=st.checkbox("Phone Service")
Multiple_lines=st.checkbox("Multiple Lines")
InternetService_Fiber_optic=st.checkbox("InternetService_Fiber optic")
OnlineSecurity=st.checkbox("Online Security")
OnlineBackup=st.checkbox("OnlineBackup_Yes")
DeviceProtection=st.checkbox("DeviceProtection")
tech_support=st.checkbox("Tech Support")
Streaming_tv=st.checkbox("Streaming TV")
streaming_movies=st.checkbox("Streaming movies")
contract=st.selectbox("Contract Type",["Select","Month-to-month","One Year","Two Year"])
PaperlessBilling=st.checkbox("Paperles Billing")
PaymentMethod=st.selectbox("Payment Method",["Select","Credit card","Electronic Check","Mailed Check"])

ip={'SeniorCitizen':int(sc), 'tenure':tenure, 'MonthlyCharges':monthly_charges, 'TotalCharges':total_charges , 'gender_Male':int(gender=="Male"), 
    'Partner_Yes':int(partner), 'Dependents_Yes':int(Dependents), 'PhoneService_Yes':int(Phone_service), 'MultipleLines_Yes':int(Multiple_lines), 
    'InternetService_Fiber optic':int(InternetService_Fiber_optic), 'InternetService_No':int(not(InternetService_Fiber_optic)), 'OnlineSecurity_Yes':int(OnlineSecurity), 
    'OnlineBackup_Yes':int(OnlineBackup), 'DeviceProtection_Yes':int(DeviceProtection), 'TechSupport_Yes':int(tech_support), 'StreamingTV_Yes':int(Streaming_tv), 
    'StreamingMovies_Yes':int(streaming_movies), 'Contract_One year':int(contract=="One Year"), 'Contract_Two year':int(contract=="Two Year"), 
    'PaperlessBilling_Yes':int(PaperlessBilling), 'PaymentMethod_Credit card (automatic)':int(PaymentMethod=="Credit Card"), 
    'PaymentMethod_Electronic check':int(PaymentMethod=="Electronic Check"), 'PaymentMethod_Mailed check':int(PaymentMethod=="Mailed Check")}

input_df=pd.DataFrame([ip])

st.write("Your Input (Check before Predicting)")
st.write(input_df)

input_scaled=scaler.transform(input_df)

st.write("Scaled Input")
st.write(input_scaled)


if st.button("Predict"):
    st.write("Prediction button clicked!")
    prediction=model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.write(
    f"Churn Probability: {probability:.2%}"
    )

    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")




