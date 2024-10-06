# Importing Libraries
from forUI import pipe
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title='House Price Prediction')
st.title('House Price Predicition')
num_bedroom = st.number_input('Enter Number of Bed Rooms',min_value=1)
num_bathroom = st.number_input('Enter Number of Bath Rooms',min_value=1)
sqrfoot = st.number_input('Enter Square Foot Are',min_value=500)
lot_size = st.number_input('Enter Lot Size',min_value=1000)
age = st.number_input("Enter Age of House",min_value=0)
prox_city = st.number_input('Enter Proximity to City Center',min_value=1)
nbr_quality = st.number_input('Enter Neighborhood Quality',min_value=1,max_value=10)

def predict():
    row = np.array([num_bedroom,num_bathroom,sqrfoot,lot_size,age,prox_city,nbr_quality])
    X = pd.DataFrame([row])
    prediction = pipe.predict(X)
    # st.success("Predicted House Price: {}".format(prediction))
    st.write("Predicted House Price is: {}$".format(str(prediction)))
    st.toast('Price is updated at top')

st.button("Get House Price",on_click=predict)