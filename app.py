import numpy as np
import pandas as pd
import streamlit as st
import pickle



html_temp = """
    
    <h1 style="color:#2e8b57;text-align:center;">🏡 Real Estate Price Predictor in Banglore</h1>
    </div>
    """

st.markdown(html_temp, unsafe_allow_html=True)

pickle_in = open('lr_clf.pkl', 'rb') 
lr_clf = pickle.load(pickle_in) 

column_pkl = open('columns.pkl', 'rb')
columns = pickle.load(column_pkl)

location = st.text_input(
    "Enter location in Bangalore:", 
    value='', 
    placeholder="Location", 
    key="location"
)
total_sqft = st.number_input(
    "Enter the area (in sq ft):", 
    min_value=100, 
    step=10, 
    key="total_sqft",
    value=100
)
bath = st.number_input(
    "Enter the number of bathrooms:", 
    min_value=1,
    step=1, 
    key="bath",
    value=1
)
bhk = st.number_input(
    "Enter the number of bedrooms:", 
    min_value=1, 
    step=1, 
    key="bhk",
    value=1
)
balcony = st.number_input(
    "Enter the number of balconies:", 
    min_value=1, 
    step=1, 
    key="balcony",
    value=1
)


if st.button("Predict Price"):
    try:
        input_data = pd.DataFrame(
            [[location, total_sqft, bath, bhk, balcony]],
            columns=["location", "total_sqft", "bath", "bhk", "balcony"]
        )

        input_data_dummies = pd.get_dummies(input_data["location"])
        input_data = input_data.drop(columns=["location"]).join(input_data_dummies)

        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data.reindex(columns=columns, fill_value=0)
        prediction = lr_clf.predict(input_data)

        st.success(f"Predicted price: ₹ {prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Error in input: {e}")
