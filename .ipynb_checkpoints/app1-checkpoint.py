import streamlit as st
import joblib
import numpy as np

# Titre et introduction
st.title("DAP/TSP Price Prediction App")
st.divider()
st.write("This app uses machine learning to predict DAP and TSP prices based on raw material prices. "
         "To use it, select the type, enter the inputs and press the Predict button.")

st.divider()

# Sélection du type de prédiction
prediction_type = st.selectbox("Choose the prediction type:", ["DAP", "TSP"])
st.divider()

# Entrée des variables
Soufre = st.number_input("Prix Soufre", min_value=0, value=0)
NH3 = st.number_input("Prix NH3", min_value=0, value=0)
ACS = st.number_input("Prix ACS", min_value=0, value=0)
st.divider()

# Préparation des données
X = [Soufre, NH3, ACS]

# Chargement des modèles
model = joblib.load("model.pkl")        # DAP
model_tsp = joblib.load("model_tsp.pkl")  # TSP

# Prédiction
predictbutton = st.button("Predict!")

if predictbutton:
    st.success("Prediction completed successfully.")
    X_array = np.array([X])  # Le modèle attend un tableau 2D

    if prediction_type == "DAP":
        prediction_dap = model.predict(X_array)
        st.markdown(f"- **Predicted DAP Price**: {float(prediction_dap[0]):,.2f}")

    elif prediction_type == "TSP":
        prediction_tsp = model_tsp.predict(X_array)
        st.markdown(f"- **Predicted TSP Price**: {float(prediction_tsp[0]):,.2f}")
else:
    st.write("ℹ️ Please enter values and click Predict.")
