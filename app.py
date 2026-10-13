import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el escalador
modelo = joblib.load('modelo_gasolina.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Predicción del Precio de la Gasolina")
st.write("Ingresa los valores de las variables para predecir el precio en USD por galón.")

# Crear campos para que el usuario ingrese los datos
# Ajusta los valores por defecto (value) según tus datos reales
precio_petroleo = st.number_input("Precio del Petróleo", value=80.0)
tipo_cambio = st.number_input("Tipo de Cambio USD/COP", value=4000.0)
demanda_diaria = st.number_input("Demanda Diaria", value=1000000.0)
prod_opep = st.number_input("Producción OPEP", value=30000000.0)
inflacion = st.number_input("Inflación (%)", value=5.0)
pib = st.number_input("PIB", value=300.0)
eventos_geopoliticos = st.selectbox("Eventos Geopolíticos (0=No, 1=Sí)", [0, 1])
impuestos = st.number_input("Porcentaje de Impuestos (%)", value=15.0)

if st.button("Predecir Precio"):
    # Organizar los datos en un DataFrame (debe coincidir con las columnas del entrenamiento)
    datos_entrada = pd.DataFrame([[
        precio_petroleo, tipo_cambio, demanda_diaria, prod_opep, 
        inflacion, pib, eventos_geopoliticos, impuestos
    ]], columns=[
       'Precio_Petroleo_USD_Barril', 'Tipo_Cambio_USD_COP', 'Demanda_miles_barriles_dia', 'Produccion_OPEP_millones_dia', 'Inflacion_Porcentaje', 'PIB_Porcentaje', 'Eventos_Geopoliticos', 'Impuestos_Combustibles_Porcentaje'
    ])
    
    # Escalar los datos y predecir
    datos_escalados = scaler.transform(datos_entrada)
    prediccion = modelo.predict(datos_escalados)[0]
    
    st.success(f"El precio estimado de la gasolina es: **${prediccion:.2f} USD/galón**")
