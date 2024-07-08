                    # Frontend Application of Price Prediction Model
# import streamlit
import streamlit as st
from joblib import load
import numpy as np

# Load the pre-trained model
model = load('Model.joblib')

features = np.array([[-0.35082247, -0.49148409, -1.25183452, -0.27144836, -0.22208489,
        0.35869844, -0.25450164,  1.07434355, -0.91150503,  0.17974455,
       -1.08091535,  0.2995192 , -0.51357301,  0.02300235]])

# Define a function to make predictions
def predict_house_price(features):
    features = np.array([features])
    price = model.predict(features) #Target_variable
    return price[0]


# Streamlit UI
st.title("House Price Prediction")

# Center content container
with st.container():
    st.markdown('<div class="center-content">', unsafe_allow_html=True)

    # Input Features
    CRIM = st.number_input("Per Capita Crime Rate By Town", value=-0.35082247)
    ZN = st.number_input("Proportion Of Residential Land Zoned For Lots Over 25,000 Sq.Ft.", value=-0.49148409)
    INDUS = st.number_input("Proportion Of Non-Retail Business Acres Per Town", value=-1.25183452)
    CHAS = st.number_input("Charles River Variable", min_value=0, max_value=1, value=0)
    NOX = st.number_input("Nitric Oxides Concentration (Parts Per 10 Million)", value=-0.22208489)
    RM = st.number_input("Average Number Of Rooms Per Dwelling", value=0.35869844)
    AGE = st.number_input("Proportion Of Owner-Occupied Units Built Prior To 1940", value=-0.25450164)
    DIS = st.number_input("Weighted Distances To Five Boston Employment Centres", value=1.07434355)
    RAD = st.number_input("Index Of Accessibility To Radial Highways", value=-0.91150503)
    TAX = st.number_input("Full-Value Property-Tax Rate Per $10,000", value=0.17974455)
    PTRATIO = st.number_input("Pupil-Teacher Ratio By Town", value=-1.08091535)
    B = st.number_input("Proportion Of Blacks By Town", value=0.2995192)
    LSTAT = st.number_input("% Lower Status Of The Population", value=-0.51357301)
    TAXRATIO = st.number_input("TAX RATIO", value=0.02300235)

    features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, TAXRATIO]

    # Button
    if st.button("Predict"):
        price = predict_house_price(features)
        st.write(f"Predicted Median Value of House: ${price:.3f}k")
    
    st.markdown('</div>', unsafe_allow_html=True)
