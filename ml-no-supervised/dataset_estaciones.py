"""
GENERADOR DE DATASET - Patrones de Uso del Metro de Medellin
==============================================================
Genera un dataset sintetico de patrones de uso de estaciones
para aplicar tecnicas de agrupamiento (clustering).

A diferencia del dataset supervisado (Actividad 3), este dataset
NO tiene variable objetivo (etiqueta). El algoritmo de clustering
debe descubrir los grupos por si solo.

Referencia:
- Palma Mendez (2008), Cap. 16: Tecnicas de agrupamiento

Cada registro representa una estacion del Metro con metricas
de uso: flujo de pasajeros, frecuencia, conectividad, etc.
"""

import csv
import random
import math

random.seed(42)

# ============================================================
# Datos del sistema de transporte
# ============================================================

ESTACIONES_INFO = {
    # Linea A
    "Niquia":          {"linea": "A", "tipo": "terminal",  "zona": "norte",    "lat": 6.3382, "lon": -75.5440},
    "Bello":           {"linea": "A", "tipo": "regular",   "zona": "norte",    "lat": 6.3342, "lon": -75.5560},
    "Madera":          {"linea": "A", "tipo": "regular",   "zona": "norte",    "lat": 6.3270, "lon": -75.5530},
    "Acevedo":         {"linea": "A", "tipo": "transbordo","zona": "norte",    "lat": 6.3190, "lon": -75.5510},
    "Tricentenario":   {"linea": "A", "tipo": "regular",   "zona": "norte",    "lat": 6.3090, "lon": -75.5530},
    "Caribe":          {"linea": "A", "tipo": "regular",   "zona": "centro",   "lat": 6.2980, "lon": -75.5540},
    "Universidad":     {"linea": "A", "tipo": "regular",   "zona": "centro",   "lat": 6.2690, "lon": -75.5660},
    "Hospital":        {"linea": "A", "tipo": "transbordo","zona": "centro",   "lat": 6.2610, "lon": -75.5700},
    "Prado":           {"linea": "A", "tipo": "regular",   "zona": "centro",   "lat": 6.2530, "lon": -75.5690},
    "Parque Berrio":   {"linea": "A", "tipo": "regular",   "zona": "centro",   "lat": 6.2490, "lon": -75.5680},
    "San Antonio":     {"linea": "A", "tipo": "transbordo","zona": "centro",   "lat": 6.2460, "lon": -75.5690},
    "Alpujarra":       {"linea": "A", "tipo": "regular",   "zona": "centro",   "lat": 6.2420, "lon": -75.5710},
    "Exposiciones":    {"linea": "A", "tipo": "regular",   "zona": "centro",   "lat": 6.2370, "lon": -75.5740},
    "Industriales":    {"linea": "A", "tipo": "transbordo","zona": "sur",      "lat": 6.2310, "lon": -75.5770},
    "Poblado":         {"linea": "A", "tipo": "regular",   "zona": "sur",      "lat": 6.2100, "lon": -75.5780},
    "Aguacatala":      {"linea": "A", "tipo": "regular",   "zona": "sur",      "lat": 6.1990, "lon": -75.5790},
    "Ayura":           {"linea": "A", "tipo": "regular",   "zona": "sur",      "lat": 6.1900, "lon": -75.5810},
    "Envigado":        {"linea": "A", "tipo": "regular",   "zona": "sur",      "lat": 6.1750, "lon": -75.5830},
    "Itagui":          {"linea": "A", "tipo": "regular",   "zona": "sur",      "lat": 6.1640, "lon": -75.5850},
    "Sabaneta":        {"linea": "A", "tipo": "regular",   "zona": "sur",      "lat": 6.1510, "lon": -75.6170},
    "La Estrella":     {"linea": "A", "tipo": "terminal",  "zona": "sur",      "lat": 6.1400, "lon": -75.6300},
    # Linea B
    "Cisneros":        {"linea": "B", "tipo": "regular",   "zona": "centro",   "lat": 6.2480, "lon": -75.5730},
    "Suramericana":    {"linea": "B", "tipo": "regular",   "zona": "occidente","lat": 6.2490, "lon": -75.5790},
    "Estadio":         {"linea": "B", "tipo": "regular",   "zona": "occidente","lat": 6.2530, "lon": -75.5870},
    "Floresta":        {"linea": "B", "tipo": "regular",   "zona": "occidente","lat": 6.2560, "lon": -75.5930},
    "Santa Lucia":     {"linea": "B", "tipo": "regular",   "zona": "occidente","lat": 6.2570, "lon": -75.5990},
    "San Javier":      {"linea": "B", "tipo": "transbordo","zona": "occidente","lat": 6.2570, "lon": -75.6120},
    # Linea K
    "Andalucia":       {"linea": "K", "tipo": "regular",   "zona": "noreste",  "lat": 6.3240, "lon": -75.5430},
    "Popular":         {"linea": "K", "tipo": "regular",   "zona": "noreste",  "lat": 6.3280, "lon": -75.5370},
    "Santo Domingo":   {"linea": "K", "tipo": "transbordo","zona": "noreste",  "lat": 6.3330, "lon": -75.5300},
    # Linea J
    "Juan XXIII":      {"linea": "J", "tipo": "regular",   "zona": "occidente","lat": 6.2680, "lon": -75.6170},
    "Vallejuelos":     {"linea": "J", "tipo": "regular",   "zona": "occidente","lat": 6.2750, "lon": -75.6210},
    "La Aurora":       {"linea": "J", "tipo": "terminal",  "zona": "occidente","lat": 6.2820, "lon": -75.6250},
    # Tranvia T-A
    "San Jose":        {"linea": "T-A","tipo": "regular",  "zona": "centro",   "lat": 6.2470, "lon": -75.5650},
    "Pabellon del Agua":{"linea":"T-A","tipo": "regular",  "zona": "centro",   "lat": 6.2475, "lon": -75.5610},
    "Bicentenario":    {"linea": "T-A","tipo": "regular",  "zona": "centro",   "lat": 6.2480, "lon": -75.5570},
    "Buenos Aires":    {"linea": "T-A","tipo": "regular",  "zona": "oriente",  "lat": 6.2485, "lon": -75.5530},
    "Miraflores":      {"linea": "T-A","tipo": "transbordo","zona": "oriente", "lat": 6.2490, "lon": -75.5490},
    "Loyola":          {"linea": "T-A","tipo": "regular",  "zona": "oriente",  "lat": 6.2495, "lon": -75.5450},
    "Alejandro Echavarria":{"linea":"T-A","tipo":"regular", "zona": "oriente", "lat": 6.2500, "lon": -75.5410},
    "Oriente":         {"linea": "T-A","tipo": "terminal",  "zona": "oriente", "lat": 6.2505, "lon": -75.5370},
    # Linea H
    "13 de Noviembre":  {"linea": "H", "tipo": "terminal", "zona": "oriente",  "lat": 6.2510, "lon": -75.5450},
    # Linea L
    "Arvi":            {"linea": "L", "tipo": "terminal",  "zona": "noreste",  "lat": 6.3600, "lon": -75.5100},
}


def generar_metricas_estacion(nombre, info):
    """
    Genera metricas de uso para una estacion.
    Las metricas dependen del tipo y zona para crear clusters naturales.
    """
    tipo = info["tipo"]
    zona = info["zona"]
    linea = info["linea"]

    # --- Pasajeros diarios promedio ---
    if tipo == "transbordo":
        pasajeros_dia = random.randint(15000, 45000)
    elif tipo == "terminal":
        pasajeros_dia = random.randint(8000, 25000)
    else:
        if zona == "centro":
            pasajeros_dia = random.randint(10000, 30000)
        elif zona in ["norte", "sur"]:
            pasajeros_dia = random.randint(5000, 18000)
        else:
            pasajeros_dia = random.randint(2000, 10000)

    # --- Proporcion hora pico (%) ---
    if zona == "centro":
        prop_pico = round(random.uniform(55, 75), 1)
    elif tipo == "terminal":
        prop_pico = round(random.uniform(60, 80), 1)
    else:
        prop_pico = round(random.uniform(35, 60), 1)

    # --- Numero de conexiones ---
    if tipo == "transbordo":
        conexiones = random.randint(3, 6)
    elif tipo == "terminal":
        conexiones = random.randint(1, 2)
    else:
        conexiones = 2

    # --- Tiempo promedio de espera (minutos) ---
    if linea in ["K", "J", "H", "L"]:
        tiempo_espera = round(random.uniform(6, 12), 1)
    elif tipo == "transbordo":
        tiempo_espera = round(random.uniform(2, 5), 1)
    else:
        tiempo_espera = round(random.uniform(3, 7), 1)

    # --- Indice de congestion (1-10) ---
    if tipo == "transbordo" and zona == "centro":
        congestion = round(random.uniform(7, 10), 1)
    elif tipo == "transbordo":
        congestion = round(random.uniform(5, 8), 1)
    elif zona == "centro":
        congestion = round(random.uniform(5, 9), 1)
    else:
        congestion = round(random.uniform(1, 6), 1)

    # --- Distancia al centro (km) - calculada desde San Antonio ---
    centro_lat, centro_lon = 6.2460, -75.5690
    dlat = info["lat"] - centro_lat
    dlon = info["lon"] - centro_lon
    dist_centro = round(math.sqrt(dlat**2 + dlon**2) * 111, 2)

    # --- Frecuencia de servicio (trenes/hora) ---
    if linea in ["K", "J", "H", "L"]:
        frecuencia = random.randint(6, 10)
    elif linea == "T-A":
        frecuencia = random.randint(8, 12)
    else:
        frecuencia = random.randint(10, 20)

    # --- Tasa de incidentes mensual ---
    if tipo == "transbordo":
        incidentes = round(random.uniform(3, 8), 1)
    elif pasajeros_dia > 15000:
        incidentes = round(random.uniform(2, 6), 1)
    else:
        incidentes = round(random.uniform(0.5, 3), 1)

    # --- Satisfaccion promedio (1-5) ---
    base_sat = 3.5
    if tiempo_espera > 8:
        base_sat -= 0.5
    if congestion > 7:
        base_sat -= 0.4
    if tipo == "transbordo":
        base_sat -= 0.2
    if zona in ["centro"]:
        base_sat += 0.2
    satisfaccion = round(min(5, max(1, base_sat + random.uniform(-0.5, 0.5))), 1)

    # --- Es metrocable (binario) ---
    es_cable = 1 if linea in ["K", "J", "H", "L"] else 0

    return {
        "estacion": nombre,
        "linea": linea,
        "tipo_estacion": tipo,
        "zona": zona,
        "pasajeros_dia": pasajeros_dia,
        "proporcion_hora_pico": prop_pico,
        "num_conexiones": conexiones,
        "tiempo_espera_min": tiempo_espera,
        "indice_congestion": congestion,
        "distancia_centro_km": dist_centro,
        "frecuencia_servicio": frecuencia,
        "incidentes_mes": incidentes,
        "satisfaccion_promedio": satisfaccion,
        "es_metrocable": es_cable,
    }


def generar_dataset():
    """Genera el dataset completo de metricas por estacion."""
    registros = []
    for nombre, info in ESTACIONES_INFO.items():
        registro = generar_metricas_estacion(nombre, info)
        registros.append(registro)
    return registros


def guardar_csv(registros, archivo="dataset_estaciones_metro.csv"):
    """Guarda el dataset en formato CSV."""
    if not registros:
        return
    campos = list(registros[0].keys())
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)
    print(f"  Dataset guardado: {archivo} ({len(registros)} registros)")


if __name__ == "__main__":
    print("=" * 60)
    print("  GENERADOR DE DATASET - Estaciones Metro de Medellin")
    print("  (Aprendizaje No Supervisado)")
    print("=" * 60)
    registros = generar_dataset()
    guardar_csv(registros)
    print(f"\n  Total estaciones: {len(registros)}")
    print(f"  Variables: {len(registros[0]) - 4} numericas + 4 descriptivas")
    tipos = {}
    for r in registros:
        t = r["tipo_estacion"]
        tipos[t] = tipos.get(t, 0) + 1
    print(f"  Tipos: {tipos}")
