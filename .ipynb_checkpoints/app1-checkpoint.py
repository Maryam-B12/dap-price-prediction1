import streamlit as st
import joblib
import numpy as np

# Titre et introduction
st.title("DAP Price Prediction App")
st.divider()
st.write("This app uses machine learning to predict DAP prices based on raw material prices. "
         "To use it, enter the inputs and press the Predict button.")
st.divider()

# Entrée des variables
Soufre = st.number_input("Prix Soufre", min_value=0, value=0)
NH3 = st.number_input("Prix NH3", min_value=0, value=0)
ACS = st.number_input("Prix ACS", min_value=0, value=0)

st.divider()

# Préparation des données
X = [Soufre, NH3, ACS]

# Chargement du modèle
model = joblib.load("model.pkl")  # Assure-toi que le fichier 'model.pkl' est bien dans le même dossier

# Prédiction
predictbutton = st.button("Predict!")

if predictbutton:
    st.balloons()
    X_array = np.array([X])  # Le modèle attend un tableau 2D
    prediction = model.predict(X_array)
    st.write(f"🎯 Price prediction is **{float(prediction[0]):,.2f}**")

else:
    st.write("ℹ️ Please enter values and click Predict.")
