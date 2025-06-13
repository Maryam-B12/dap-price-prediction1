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

# Entrée des variables dynamiques
if prediction_type == "DAP":
    st.info("""
    📌 To predict **The DAP price at month t**, Please enter :
    - The **Sulfur price at month t−2**
    - The **NH3** and **ACS prices at month t−1**
    """)
    Soufre = st.number_input("Prix Soufre", min_value=0, value=0)
    NH3 = st.number_input("Prix NH3", min_value=0, value=0)
    ACS = st.number_input("Prix ACS", min_value=0, value=0)
    X = [Soufre, NH3, ACS]

elif prediction_type == "TSP":
    st.info("""
    📌 To predict **the TSP price at month t**, please enter :
    - The **Sulfur price at month t−1**
    - The **ACS price at month t−2**
    """)
    Soufre = st.number_input("Prix Soufre", min_value=0, value=0)
    ACS = st.number_input("Prix ACS", min_value=0, value=0)
    X = [Soufre, ACS]  # NH3 exclu ici

st.divider()

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


import streamlit.components.v1 as components

# Rafraîchissement automatique toutes les 50 minutes (50 × 60 × 1000 ms)
components.html("""
    <script>
        setTimeout(function(){
            window.location.reload();
        }, 3000000);
    </script>
""", height=0)

