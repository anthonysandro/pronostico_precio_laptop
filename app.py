import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
#with open('price_laptop_group3', 'rb') as file:
#    #modelo = pickle.load(file)
modelo = joblib.load('price_laptop_group3.pkl')
# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')

# Controles de entrada para las características
# 'GHz', 'Ram', 'screen_width', 'screen_height']
ghz = st.number_input('GHz del CPU', min_value=0.1, max_value=5.0, value=2.5)
ram = st.number_input('RAM (GB)', min_value=1, max_value=64, value=8)
screen_width = st.number_input('Ancho de Pantalla', min_value=800, max_value=4000, value=1920)
screen_height = st.number_input('Alto de Pantalla', min_value=600, max_value=3000, value=1080)

type_gaming = st.selectbox('¿Es Gaming?', ['No', 'Sí'])
type_notebook = st.selectbox('¿Es Notebook?', ['No', 'Sí'])

# Convertir entradas a formato numérico
type_gaming = 1 if type_gaming == 'Sí' else 0
type_notebook = 1 if type_notebook == 'Sí' else 0

# Botón para realizar predicción
if st.button('Predict Price'):
    # Crear DataFrame con las entradas
    #input_data = pd.DataFrame([[ghz, ram, screen_width, screen_height,  type_gaming, type_notebook]],
    #                columns=['GHz','Ram', 'screen_width', 'screen_height', 'TypeName_Gaming', 'TypeName_Notebook'])
    input_data = pd.DataFrame([[ghz, ram, screen_width, screen_height]],
                    columns=['GHz','Ram', 'screen_width', 'screen_height'])

    # Estandarización de las características
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Realizar predicción
    forecasting = modelo.predict(input_scaled)

    # Mostrar predicción
    st.write(f'Forecast the price of a laptop: {forecasting[0]:.2f} euros')