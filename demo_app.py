import streamlit as st
import requests
import json


st.title('Bimser Staj Demo App')

number1 = st.number_input('Insert a number 1', value=0)
number2 = st.number_input('Insert a number 2', value=0)

if st.button('Çalıştır'):
    payload = json.dumps({
        "val1": number1,
        "val2": number2
    })

    response = requests.request("POST", "http://127.0.0.1:8000/predict", headers={'Content-Type': 'application/json'}, data=payload)

    st.text(response.json())
