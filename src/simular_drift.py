import requests
import time
import random

URL = "http://localhost:8000/predict"

# Añadimos suficiente variedad de marcas para que Evidently calcule frecuencias categóricas
marcas_variadas = ["PORSCHE", "FERRARI", "LAMBORGHINI", "ASTON MARTIN", "AUDI", "BMW", "MERCEDES"]
usos_variados = ["Professional", "Private", "Leisure"]

print("--- Lanzando ráfaga de peticiones válidas con alta variabilidad estadística ---")

for i in range(50):
    payload = {
            "OBJECT_ID": 9000 + i,
            "INSR_BEGIN": "2026-01-01",
            "INSR_END": "2027-01-01",
            "INSR_TYPE": "1202", 
            "PROD_YEAR": 2018.0,                          
            "PREMIUM": 5000000.0,                         # <-- DRIFT: 5 millones de euros (máximo histórico era 183 mil)
            "INSURED_VALUE": 900000000.0,                 # <-- DRIFT: 900 millones de euros (máximo histórico era 9 millones)
            "SEX": "1",              
            "MAKE": "TOYOTA",                           
            "USAGE": "Private",                     
            "TYPE_VEHICLE": "Automobile",                       
            "SEATS_NUM": 150.0,                           # <-- DRIFT: 150 asientos (máximo histórico era 12)
            "CARRYING_CAPACITY": 80000.0,                 # <-- DRIFT: 80 toneladas (máximo histórico era 155)
            "CCM_TON": 250000.0,                          # <-- DRIFT: Motor gigante
            "EFFECTIVE_YR": 26                            
        }
    
    try:
        response = requests.post(URL, json=payload)
        print(f"Petición {i+1}/50 enviada. Status API: {response.status_code}")
    except Exception as e:
        print(f"Error de conexión: {e}")
        
    time.sleep(0.05) # Ejecución más rápida

print("--- Ráfaga finalizada. Esperando la reacción del Monitor ---")