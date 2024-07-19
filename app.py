import streamlit as st
import pickle
import json
import numpy as np

# Load the model and data columns
def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    with open("model\columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    with open("model\Banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

def get_location_names():
    return __locations

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# Load artifacts
load_saved_artifacts()

# Streamlit app
st.title("Bangalore Home Price Prediction")

sqft = st.text_input("Area (Square Feet)", "1000")
bhk = st.radio("BHK", [1, 2, 3, 4, 5], index=1)
bath = st.radio("Bathrooms", [1, 2, 3, 4, 5], index=1)
location = st.selectbox("Location", get_location_names())

if st.button("Estimate Price"):
    price = get_estimated_price(location, float(sqft), bhk, bath)
    st.write(f"The estimated price is {price} Lakh")
