"""
GENERADOR DE DATASET - Viajes en el Metro de Medellín
======================================================
Genera un dataset sintético de viajes para entrenar modelos
de aprendizaje supervisado (árboles de decisión).

Cada registro representa un viaje entre dos estaciones con
características extraídas del sistema de transporte y la
variable objetivo: satisfaccion_usuario (alta/media/baja).

Basado en:
- Palma Méndez (2008), Cap. 17: Árboles de decisión
- Datos del sistema Metro de Medellín (Actividades 1 y 2)
"""

import csv
import random
import math
import os

random.seed(42)

# ============================================================
# Datos del sistema de transporte (de base_conocimiento.py)
# ============================================================

ESTACIONES_LINEA = {
    "A": ["Niquía", "Bello", "Madera", "Acevedo", "Tricentenario",
          "Caribe", "Universidad", "Hospital", "Prado", "Parque Berrío",
          "San Antonio", "Alpujarra", "Exposiciones", "Industriales",
          "Poblado", "Aguacatala", "Ayurá", "Envigado", "Itagüí",
          "Sabaneta", "La Estrella"],
    "B": ["San Antonio", "Cisneros", "Suramericana", "Estadio",
          "Floresta", "Santa Lucía", "San Javier"],
    "K": ["Acevedo", "Andalucía", "Popular", "Santo Domingo"],
    "J": ["San Javier", "Juan XXIII", "Vallejuelos", "La Aurora"],
    "T-A": ["San Antonio", "San José", "Pabellón del Agua",
            "Bicentenario", "Buenos Aires", "Miraflores",
            "Loyola", "Alejandro Echavarría", "Oriente"],
    "H": ["Miraflores", "13 de Noviembre"],
    "L": ["Santo Domingo", "Arví"],
}

TRANSBORDOS = {"Acevedo", "Hospital", "San Antonio", "Industriales",
               "San Javier", "Santo Domingo", "Miraflores"}

METROCABLES = {"K", "J", "H", "L"}

TERMINALES = {"Niquía", "La Estrella", "San Javier", "Santo Domingo",
              "La Aurora", "Oriente", "13 de Noviembre", "Arví"}

# Franjas horarias y sus características
FRANJAS = {
    "pico_manana":   {"hora_inicio": 6,  "hora_fin": 9,  "congestion": "alta"},
    "valle_manana":  {"hora_inicio": 9,  "hora_fin": 12, "congestion": "baja"},
    "pico_mediodia": {"hora_inicio": 12, "hora_fin": 14, "congestion": "media"},
    "valle_tarde":   {"hora_inicio": 14, "hora_fin": 17, "congestion": "baja"},
    "pico_tarde":    {"hora_inicio": 17, "hora_fin": 20, "congestion": "alta"},
    "noche":         {"hora_inicio": 20, "hora_fin": 23, "congestion": "baja"},
}

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes",
               "sábado", "domingo"]

CLIMAS = ["soleado", "nublado", "lluvia"]


def obtener_linea_estacion(estacion):
    """Retorna la línea principal de una estación."""
    for linea, estaciones in ESTACIONES_LINEA.items():
        if estacion in estaciones:
            return linea
    return "A"


def calcular_num_estaciones(origen, destino):
    """Calcula el número aproximado de estaciones entre origen y destino."""
    linea_o = obtener_linea_estacion(origen)
    linea_d = obtener_linea_estacion(destino)
    if linea_o == linea_d:
        estaciones = ESTACIONES_LINEA[linea_o]
        if origen in estaciones and destino in estaciones:
            return abs(estaciones.index(origen) - estaciones.index(destino))
    return random.randint(5, 20)


def calcular_transbordos(origen, destino):
    """Estima el número de transbordos necesarios."""
    linea_o = obtener_linea_estacion(origen)
    linea_d = obtener_linea_estacion(destino)
    if linea_o == linea_d:
        return 0
    # Verificar si comparten transbordo directo
    for t in TRANSBORDOS:
        lo = obtener_linea_estacion(t)
        if t in ESTACIONES_LINEA.get(linea_o, []) and t in ESTACIONES_LINEA.get(linea_d, []):
            return 1
    return random.choice([2, 3])


def generar_tiempo_viaje(num_estaciones, num_transbordos, clima, usa_cable, franja):
    """Genera tiempo de viaje realista basado en las reglas del sistema."""
    # Tiempo base: ~2.5 min por estación
    tiempo = num_estaciones * 2.5
    # Penalización por transbordo: +3 min (Regla R5)
    tiempo += num_transbordos * 3.0
    # Penalización por lluvia en cable: +5 min (Regla R7)
    if clima == "lluvia" and usa_cable:
        tiempo += 5.0
    # Penalización por congestión
    info_franja = FRANJAS.get(franja, {"congestion": "baja"})
    if info_franja["congestion"] == "alta":
        tiempo *= 1.15
    elif info_franja["congestion"] == "media":
        tiempo *= 1.08
    # Ruido aleatorio
    tiempo += random.uniform(-1.5, 2.0)
    return round(max(tiempo, 3.0), 1)


def determinar_satisfaccion(tiempo_viaje, num_transbordos, clima,
                             congestion, usa_cable, dia):
    """
    Determina la satisfacción del usuario basándose en reglas
    que un árbol de decisión debería aprender.

    Reglas principales:
    - Tiempo < 15 min y sin transbordos -> alta
    - Tiempo > 45 min o (lluvia + cable) -> baja
    - 2+ transbordos con congestión alta -> baja
    - Fin de semana con buen clima -> tiende a alta
    - Resto -> media
    """
    score = 50  # base

    # Factor tiempo
    if tiempo_viaje < 10:
        score += 30
    elif tiempo_viaje < 20:
        score += 15
    elif tiempo_viaje < 30:
        score += 0
    elif tiempo_viaje < 45:
        score -= 15
    else:
        score -= 30

    # Factor transbordos
    if num_transbordos == 0:
        score += 15
    elif num_transbordos == 1:
        score += 0
    elif num_transbordos == 2:
        score -= 10
    else:
        score -= 20

    # Factor clima
    if clima == "soleado":
        score += 10
    elif clima == "lluvia":
        score -= 15
        if usa_cable:
            score -= 10  # Penalización extra

    # Factor congestión
    if congestion == "alta":
        score -= 15
    elif congestion == "baja":
        score += 10

    # Factor día
    if dia in ["sábado", "domingo"]:
        score += 10

    # Ruido
    score += random.randint(-8, 8)

    # Clasificación
    if score >= 65:
        return "alta"
    elif score >= 40:
        return "media"
    else:
        return "baja"


def generar_dataset(n_registros=500):
    """Genera el dataset completo de viajes."""
    todas_estaciones = []
    for linea, ests in ESTACIONES_LINEA.items():
        for e in ests:
            if e not in [x for x in todas_estaciones]:
                todas_estaciones.append(e)

    registros = []

    for i in range(n_registros):
        # Seleccionar origen y destino diferentes
        origen = random.choice(todas_estaciones)
        destino = random.choice(todas_estaciones)
        while destino == origen:
            destino = random.choice(todas_estaciones)

        linea_origen = obtener_linea_estacion(origen)
        linea_destino = obtener_linea_estacion(destino)
        misma_linea = 1 if linea_origen == linea_destino else 0

        num_estaciones = calcular_num_estaciones(origen, destino)
        num_transbordos = calcular_transbordos(origen, destino)

        usa_cable = 1 if (linea_origen in METROCABLES or
                          linea_destino in METROCABLES) else 0

        origen_terminal = 1 if origen in TERMINALES else 0
        destino_terminal = 1 if destino in TERMINALES else 0

        # Condiciones del viaje
        clima = random.choices(CLIMAS, weights=[0.5, 0.3, 0.2])[0]
        dia = random.choice(DIAS_SEMANA)
        es_fin_semana = 1 if dia in ["sábado", "domingo"] else 0
        franja = random.choice(list(FRANJAS.keys()))
        congestion = FRANJAS[franja]["congestion"]

        # Tiempo de viaje
        tiempo_viaje = generar_tiempo_viaje(
            num_estaciones, num_transbordos, clima,
            usa_cable == 1, franja
        )

        # Variable objetivo
        satisfaccion = determinar_satisfaccion(
            tiempo_viaje, num_transbordos, clima,
            congestion, usa_cable == 1, dia
        )

        registros.append({
            "id_viaje": i + 1,
            "origen": origen,
            "destino": destino,
            "linea_origen": linea_origen,
            "linea_destino": linea_destino,
            "misma_linea": misma_linea,
            "num_estaciones": num_estaciones,
            "num_transbordos": num_transbordos,
            "usa_metrocable": usa_cable,
            "origen_es_terminal": origen_terminal,
            "destino_es_terminal": destino_terminal,
            "dia_semana": dia,
            "es_fin_semana": es_fin_semana,
            "franja_horaria": franja,
            "congestion": congestion,
            "clima": clima,
            "tiempo_viaje_min": tiempo_viaje,
            "satisfaccion_usuario": satisfaccion,
        })

    return registros


def guardar_csv(registros, archivo="dataset_viajes_metro.csv"):
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
    print("  GENERADOR DE DATASET - Metro de Medellín")
    print("=" * 60)
    registros = generar_dataset(500)
    guardar_csv(registros)

    # Estadísticas
    alta = sum(1 for r in registros if r["satisfaccion_usuario"] == "alta")
    media = sum(1 for r in registros if r["satisfaccion_usuario"] == "media")
    baja = sum(1 for r in registros if r["satisfaccion_usuario"] == "baja")
    print(f"\n  Distribución de clases:")
    print(f"    Alta:  {alta} ({alta/len(registros)*100:.1f}%)")
    print(f"    Media: {media} ({media/len(registros)*100:.1f}%)")
    print(f"    Baja:  {baja} ({baja/len(registros)*100:.1f}%)")
