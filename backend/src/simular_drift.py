# import requests
# import time
# import random

# URL = "http://localhost:8000/predict"

# # Añadimos suficiente variedad de marcas para que Evidently calcule frecuencias categóricas
# marcas_variadas = ["PORSCHE", "FERRARI", "LAMBORGHINI", "ASTON MARTIN", "AUDI", "BMW", "MERCEDES"]
# usos_variados = ["Professional", "Private", "Leisure"]

# print("--- Lanzando ráfaga de peticiones válidas con alta variabilidad estadística ---")

# for i in range(50):
#     payload = {
#             "OBJECT_ID": random.randint(1, 1000),
#             "INSR_BEGIN": "2026-01-01",
#             "INSR_END": "2027-01-01",
#             "INSR_TYPE": "1202", 
#             "PROD_YEAR": 2018.0,                          
#             "PREMIUM": 5000000.0,                         # <-- DRIFT: 5 millones de euros (máximo histórico era 183 mil)
#             "INSURED_VALUE": 900000000.0,                 # <-- DRIFT: 900 millones de euros (máximo histórico era 9 millones)
#             "SEX": "1",              
#             "MAKE": random.choice(marcas_variadas),                           
#             "USAGE": random.choice(usos_variados),                     
#             "TYPE_VEHICLE": "Automobile",                       
#             "SEATS_NUM": 150.0,                           # <-- DRIFT: 150 asientos (máximo histórico era 12)
#             "CARRYING_CAPACITY": 80000.0,                 # <-- DRIFT: 80 toneladas (máximo histórico era 155)
#             "CCM_TON": 250000.0,                          # <-- DRIFT: Motor gigante
#             "EFFECTIVE_YR": 26                            
#         }
    
#     try:
#         response = requests.post(URL, json=payload)
#         print(f"Petición {i+1}/50 enviada. Status API: {response.status_code}")
#     except Exception as e:
#         print(f"Error de conexión: {e}")
        
#     time.sleep(0.05) # Ejecución más rápida

# print("--- Ráfaga finalizada. Esperando la reacción del Monitor ---")


import requests
import time
import random

URL = "http://localhost:8000/predict"

# Listas de valores normales para simular datos cotidianos sin drift
marcas_normales = ["TOYOTA", "RENAULT", "FORD", "NISSAN", "SEAT"]
usos_normales = ["Private", "Professional"]

print("--- Lanzando ráfaga con ~80% de Data Drift (algunas variables se mantienen normales) ---")

for i in range(50):
    payload = {
        # 1. VARIABLES NORMALES (Sin Drift): Para bajar el porcentaje total
        "OBJECT_ID": random.randint(1, 500),              # ID en rango normal
        "PROD_YEAR": float(random.randint(2010, 2020)),   # Años de fabricación comunes
        "MAKE": random.choice(marcas_normales),           # Marcas habituales
        "USAGE": random.choice(usos_normales),            # Usos típicos
        "SEX": random.choice(["1", "2"]),                 # Género estándar
        "TYPE_VEHICLE": "Automobile",                     # Tipo común
        
        # 2. VARIABLES CON DRIFT REALISTA: Valores masivos pero con variedad
        "PREMIUM": float(random.randint(4000000, 6000000)),         
        "INSURED_VALUE": float(random.randint(800000000, 1000000000)), 
        "SEATS_NUM": float(random.randint(100, 150)),               
        "CARRYING_CAPACITY": float(random.randint(70000, 90000)),     
        "CCM_TON": float(random.randint(200000, 300000)),              
        
        # Fechas fijas (pueden o no generar drift según tu histórico, actúan como control)
        "INSR_BEGIN": "2026-01-01",
        "INSR_END": "2027-01-01",
        "INSR_TYPE": "1202",
        "EFFECTIVE_YR": 26                            
    }
    
    try:
        response = requests.post(URL, json=payload)
        print(f"Petición {i+1}/50 enviada. Status API: {response.status_code}")
    except Exception as e:
        print(f"Error de conexión: {e}")
        
    time.sleep(0.05)

print("--- Ráfaga finalizada. Esperando la reacción del Monitor ---")