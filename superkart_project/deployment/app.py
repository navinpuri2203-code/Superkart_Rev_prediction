import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="navinpuri2203/superkart-prediction-model", filename="superkart_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Superkart Prediction")
st.write("Fill the details of the Products and Store types to find the most demanding choice")

# Collect user input

ProductID = st.selectbox("Product Id", ["FD6114", "FD7839", "FD5075"])
Productweight = st.slider("Product Weight", 12.66, 16.54, 14.28)
ProductMRP = st.slider("Product MRP", 117.08, 171.43, 162.08)
StoreId = st.selectbox("Store Id",["OUT004", "OUT003", "OUT002", "OUT001"])
StoreEstablishmentyear = st.slider("Store Establishment_Year", 2009, 1999, 1987, 1998)
StoreSize = st.selectbox("Store Size", ["Medium", "High", "Small"])
ProductSugarContent = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "Low Sugar", "No Sugar"])
ProductAllocatedArea = st.slider("Product Allocated Area", 0.027, 0.144, 0.031, 0.112)
ProductType = st.selectbox("Product Type",["Frozen Foods", "Dairy", "Canned", "Baking Goods", "Snack Foods", "Meat", "Household"])
StoreLocationCityType = st.selectbox("Store Location City Type",["Tier 1", "Tier 2", "Tier 3"])
StoreType = st.selectbox("Store Type",["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
ProductStoreSalesTotal = st.slider("Product Store Sales Total", 2842.4, 4830.02, 4130.16, 4132.18)


# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{

  'ProductID': ProductID,
  'Productweight': Productweight,
  'ProductMRP': ProductMRP,
  'StoreId': StoreId,
  'StoreEstablishmentyear': StoreEstablishmentyear,
  'StoreSize': StoreSize,
  'ProductSugarContent': ProductSugarContent,
  'ProductAllocatedArea': ProductAllocatedArea,
  'ProductType': ProductType,
  'StoreLocationCityType': StoreLocationCityType,
  'StoreType': StoreType,
  'ProductStoreSalesTotal': ProductStoreSalesTotal

}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0,1]
    pred = int(prob >= classification_threshold)
    result = "will increase the revenue" if pred == 1 else "is unlikely to increase sales"
    st.write(f"Prediction: Product {result}")
