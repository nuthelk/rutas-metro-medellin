"""
BASE DE CONOCIMIENTO - Sistema de Transporte Masivo de Medellín
================================================================
Representación del conocimiento mediante reglas lógicas y hechos
sobre el sistema Metro de Medellín (Metro, Tranvía, Cables).

Referencias:
- Benítez, R. (2014). Inteligencia artificial avanzada. Cap. 2 y 3.
  Lógica y representación del conocimiento / Sistemas basados en reglas.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math


# ============================================================
# HECHOS: Estaciones y sus propiedades
# ============================================================

@dataclass
class Estacion:
    """Hecho: representa una estación del sistema de transporte."""
    nombre: str
    linea: str
    coordenadas: Tuple[float, float]  # (latitud, longitud) aprox.
    es_terminal: bool = False
    es_transbordo: bool = False
    lineas_conectadas: List[str] = field(default_factory=list)


@dataclass
class Conexion:
    """Hecho: conexión directa entre dos estaciones."""
    origen: str
    destino: str
    linea: str
    tiempo_minutos: float  # tiempo estimado de viaje
    distancia_km: float    # distancia aproximada


@dataclass
class Regla:
    """Representación de una regla lógica en la base de conocimiento."""
    nombre: str
    condiciones: List[str]
    conclusion: str
    prioridad: int = 1


class BaseConocimiento:
    """
    Base de conocimiento del sistema de transporte masivo de Medellín.
    
    Contiene:
    - Hechos: estaciones, conexiones, propiedades
    - Reglas lógicas: para inferencia sobre rutas y transbordos
    """

    def __init__(self):
        self.estaciones: Dict[str, Estacion] = {}
        self.conexiones: List[Conexion] = []
        self.reglas: List[Regla] = []
        self.grafo: Dict[str, List[Tuple[str, float, str]]] = {}
        self._cargar_estaciones()
        self._cargar_conexiones()
        self._cargar_reglas()
        self._construir_grafo()

    # --------------------------------------------------------
    # HECHOS: Estaciones del sistema
    # --------------------------------------------------------
    def _cargar_estaciones(self):
        """Carga los hechos sobre estaciones del sistema Metro de Medellín."""

        # === LÍNEA A (Niquía - La Estrella) ===
        linea_a = [
            ("Niquía",          (6.3382, -75.5440), True,  False),
            ("Bello",           (6.3342, -75.5560), False, False),
            ("Madera",          (6.3270, -75.5530), False, False),
            ("Acevedo",         (6.3190, -75.5510), False, True),  # Transbordo Línea K
            ("Tricentenario",   (6.3090, -75.5530), False, False),
            ("Caribe",          (6.2980, -75.5540), False, False),
            ("Universidad",     (6.2690, -75.5660), False, False),
            ("Hospital",        (6.2610, -75.5700), False, True),  # Transbordo Línea B
            ("Prado",           (6.2530, -75.5690), False, False),
            ("Parque Berrío",   (6.2490, -75.5680), False, False),
            ("San Antonio",     (6.2460, -75.5690), False, True),  # Transbordo Línea B y Tranvía
            ("Alpujarra",       (6.2420, -75.5710), False, False),
            ("Exposiciones",    (6.2370, -75.5740), False, False),
            ("Industriales",    (6.2310, -75.5770), False, True),  # Transbordo Línea J
            ("Poblado",         (6.2100, -75.5780), False, False),
            ("Aguacatala",      (6.1990, -75.5790), False, False),
            ("Ayurá",           (6.1900, -75.5810), False, False),
            ("Envigado",        (6.1750, -75.5830), False, False),
            ("Itagüí",          (6.1640, -75.5850), False, False),
            ("Sabaneta",        (6.1510, -75.6170), False, False),
            ("La Estrella",     (6.1400, -75.6300), True,  False),
        ]

        for nombre, coord, terminal, transbordo in linea_a:
            lineas = ["A"]
            if nombre == "Acevedo":
                lineas.append("K")
            elif nombre == "Hospital":
                lineas.extend(["B"])
            elif nombre == "San Antonio":
                lineas.extend(["B", "T-A"])
            elif nombre == "Industriales":
                lineas.append("J")
            self.estaciones[nombre] = Estacion(
                nombre=nombre, linea="A", coordenadas=coord,
                es_terminal=terminal, es_transbordo=transbordo,
                lineas_conectadas=lineas
            )

        # === LÍNEA B (San Antonio - San Javier) ===
        linea_b = [
            # San Antonio ya está definida
            ("Cisneros",        (6.2480, -75.5730), False, False),
            ("Suramericana",    (6.2490, -75.5790), False, False),
            ("Estadio",         (6.2530, -75.5870), False, False),
            ("Floresta",        (6.2560, -75.5930), False, False),
            ("Santa Lucía",     (6.2570, -75.5990), False, False),
            ("San Javier",      (6.2570, -75.6120), True,  True),  # Transbordo Línea J
        ]

        for nombre, coord, terminal, transbordo in linea_b:
            if nombre not in self.estaciones:
                lineas = ["B"]
                if nombre == "San Javier":
                    lineas.append("J")
                self.estaciones[nombre] = Estacion(
                    nombre=nombre, linea="B", coordenadas=coord,
                    es_terminal=terminal, es_transbordo=transbordo,
                    lineas_conectadas=lineas
                )

        # === LÍNEA K (Metrocable Acevedo - Santo Domingo) ===
        linea_k = [
            # Acevedo ya está definida
            ("Andalucía",       (6.3240, -75.5430), False, False),
            ("Popular",         (6.3280, -75.5370), False, False),
            ("Santo Domingo",   (6.3330, -75.5300), True,  True),  # Transbordo Línea L
        ]

        for nombre, coord, terminal, transbordo in linea_k:
            if nombre not in self.estaciones:
                lineas = ["K"]
                if nombre == "Santo Domingo":
                    lineas.append("L")
                self.estaciones[nombre] = Estacion(
                    nombre=nombre, linea="K", coordenadas=coord,
                    es_terminal=terminal, es_transbordo=transbordo,
                    lineas_conectadas=lineas
                )

        # === LÍNEA J (Metrocable San Javier - La Aurora) ===
        linea_j = [
            # San Javier ya definida
            ("Juan XXIII",      (6.2680, -75.6170), False, False),
            ("Vallejuelos",     (6.2750, -75.6210), False, False),
            ("La Aurora",       (6.2820, -75.6250), True,  False),
        ]

        for nombre, coord, terminal, transbordo in linea_j:
            if nombre not in self.estaciones:
                self.estaciones[nombre] = Estacion(
                    nombre=nombre, linea="J", coordenadas=coord,
                    es_terminal=terminal, es_transbordo=transbordo,
                    lineas_conectadas=["J"]
                )

        # === TRANVÍA LÍNEA T-A (San Antonio - Oriente) ===
        linea_ta = [
            # San Antonio ya definida
            ("San José",        (6.2470, -75.5650), False, False),
            ("Pabellón del Agua", (6.2475, -75.5610), False, False),
            ("Bicentenario",    (6.2480, -75.5570), False, False),
            ("Buenos Aires",    (6.2485, -75.5530), False, False),
            ("Miraflores",      (6.2490, -75.5490), False, True),  # Transbordo Línea H
            ("Loyola",          (6.2495, -75.5450), False, False),
            ("Alejandro Echavarría", (6.2500, -75.5410), False, False),
            ("Oriente",         (6.2505, -75.5370), True,  False),
        ]

        for nombre, coord, terminal, transbordo in linea_ta:
            if nombre not in self.estaciones:
                lineas = ["T-A"]
                if nombre == "Miraflores":
                    lineas.append("H")
                self.estaciones[nombre] = Estacion(
                    nombre=nombre, linea="T-A", coordenadas=coord,
                    es_terminal=terminal, es_transbordo=transbordo,
                    lineas_conectadas=lineas
                )

        # === LÍNEA H (Metrocable Miraflores - Trece de Noviembre) ===
        linea_h = [
            # Miraflores ya definida
            ("13 de Noviembre",  (6.2490, -75.5410), True, False),
        ]

        for nombre, coord, terminal, transbordo in linea_h:
            if nombre not in self.estaciones:
                self.estaciones[nombre] = Estacion(
                    nombre=nombre, linea="H", coordenadas=coord,
                    es_terminal=terminal, es_transbordo=transbordo,
                    lineas_conectadas=["H"]
                )

        # === LÍNEA L (Metrocable Santo Domingo - Arví) ===
        linea_l = [
            # Santo Domingo ya definida
            ("Arví",             (6.2810, -75.5020), True, False),
        ]

        for nombre, coord, terminal, transbordo in linea_l:
            if nombre not in self.estaciones:
                self.estaciones[nombre] = Estacion(
                    nombre=nombre, linea="L", coordenadas=coord,
                    es_terminal=terminal, es_transbordo=transbordo,
                    lineas_conectadas=["L"]
                )

    # --------------------------------------------------------
    # HECHOS: Conexiones entre estaciones
    # --------------------------------------------------------
    def _cargar_conexiones(self):
        """Carga los hechos sobre conexiones directas entre estaciones."""

        def _agregar_conexion(origen, destino, linea, tiempo, distancia):
            """Agrega conexión bidireccional."""
            self.conexiones.append(Conexion(origen, destino, linea, tiempo, distancia))
            self.conexiones.append(Conexion(destino, origen, linea, tiempo, distancia))

        # --- Línea A ---
        pares_a = [
            ("Niquía", "Bello", 2.5, 1.2),
            ("Bello", "Madera", 2.0, 0.9),
            ("Madera", "Acevedo", 2.0, 0.8),
            ("Acevedo", "Tricentenario", 2.0, 1.0),
            ("Tricentenario", "Caribe", 2.5, 1.1),
            ("Caribe", "Universidad", 3.0, 1.5),
            ("Universidad", "Hospital", 2.0, 0.8),
            ("Hospital", "Prado", 1.5, 0.6),
            ("Prado", "Parque Berrío", 1.5, 0.5),
            ("Parque Berrío", "San Antonio", 1.5, 0.5),
            ("San Antonio", "Alpujarra", 1.5, 0.5),
            ("Alpujarra", "Exposiciones", 2.0, 0.7),
            ("Exposiciones", "Industriales", 2.0, 0.7),
            ("Industriales", "Poblado", 3.0, 1.5),
            ("Poblado", "Aguacatala", 2.5, 1.2),
            ("Aguacatala", "Ayurá", 2.0, 0.9),
            ("Ayurá", "Envigado", 2.5, 1.1),
            ("Envigado", "Itagüí", 2.5, 1.2),
            ("Itagüí", "Sabaneta", 3.0, 1.5),
            ("Sabaneta", "La Estrella", 3.0, 1.4),
        ]
        for o, d, t, dist in pares_a:
            _agregar_conexion(o, d, "A", t, dist)

        # --- Línea B ---
        pares_b = [
            ("San Antonio", "Cisneros", 2.0, 0.6),
            ("Cisneros", "Suramericana", 2.0, 0.7),
            ("Suramericana", "Estadio", 2.0, 0.8),
            ("Estadio", "Floresta", 2.0, 0.7),
            ("Floresta", "Santa Lucía", 2.0, 0.7),
            ("Santa Lucía", "San Javier", 2.5, 1.0),
        ]
        for o, d, t, dist in pares_b:
            _agregar_conexion(o, d, "B", t, dist)

        # --- Línea K (Metrocable) ---
        pares_k = [
            ("Acevedo", "Andalucía", 4.0, 0.9),
            ("Andalucía", "Popular", 3.5, 0.7),
            ("Popular", "Santo Domingo", 3.5, 0.8),
        ]
        for o, d, t, dist in pares_k:
            _agregar_conexion(o, d, "K", t, dist)

        # --- Línea J (Metrocable) ---
        pares_j = [
            ("San Javier", "Juan XXIII", 4.0, 0.8),
            ("Juan XXIII", "Vallejuelos", 3.5, 0.6),
            ("Vallejuelos", "La Aurora", 3.5, 0.7),
        ]
        for o, d, t, dist in pares_j:
            _agregar_conexion(o, d, "J", t, dist)

        # --- Tranvía T-A ---
        pares_ta = [
            ("San Antonio", "San José", 2.5, 0.5),
            ("San José", "Pabellón del Agua", 2.0, 0.4),
            ("Pabellón del Agua", "Bicentenario", 2.0, 0.4),
            ("Bicentenario", "Buenos Aires", 2.0, 0.4),
            ("Buenos Aires", "Miraflores", 2.0, 0.4),
            ("Miraflores", "Loyola", 2.0, 0.4),
            ("Loyola", "Alejandro Echavarría", 2.0, 0.4),
            ("Alejandro Echavarría", "Oriente", 2.5, 0.5),
        ]
        for o, d, t, dist in pares_ta:
            _agregar_conexion(o, d, "T-A", t, dist)

        # --- Línea H (Metrocable) ---
        _agregar_conexion("Miraflores", "13 de Noviembre", "H", 5.0, 1.0)

        # --- Línea L (Metrocable) ---
        _agregar_conexion("Santo Domingo", "Arví", "L", 12.0, 4.5)

    # --------------------------------------------------------
    # REGLAS LÓGICAS del sistema basado en reglas
    # --------------------------------------------------------
    def _cargar_reglas(self):
        """
        Define las reglas lógicas del sistema experto.
        Basado en: Benítez (2014), Cap. 3 - Sistemas basados en reglas.
        
        Formato: SI (condiciones) ENTONCES (conclusión)
        """
        self.reglas = [
            Regla(
                nombre="R1_misma_linea",
                condiciones=["estacion_origen.linea == estacion_destino.linea"],
                conclusion="ruta_directa_sin_transbordo",
                prioridad=3
            ),
            Regla(
                nombre="R2_transbordo_simple",
                condiciones=[
                    "estacion_origen.linea != estacion_destino.linea",
                    "existe_estacion_transbordo_comun(origen, destino)"
                ],
                conclusion="ruta_con_un_transbordo",
                prioridad=2
            ),
            Regla(
                nombre="R3_transbordo_multiple",
                condiciones=[
                    "estacion_origen.linea != estacion_destino.linea",
                    "NOT existe_estacion_transbordo_comun(origen, destino)"
                ],
                conclusion="ruta_con_multiples_transbordos",
                prioridad=1
            ),
            Regla(
                nombre="R4_preferir_metro_sobre_cable",
                condiciones=[
                    "ruta_alternativa_metro_disponible",
                    "ruta_alternativa_cable_disponible",
                    "diferencia_tiempo < 5 minutos"
                ],
                conclusion="preferir_ruta_metro",
                prioridad=2
            ),
            Regla(
                nombre="R5_penalizar_transbordo",
                condiciones=["requiere_transbordo"],
                conclusion="agregar_penalizacion_tiempo(3_minutos)",
                prioridad=2
            ),
            Regla(
                nombre="R6_terminal_acceso",
                condiciones=["estacion.es_terminal == True"],
                conclusion="considerar_mayor_tiempo_espera",
                prioridad=1
            ),
            Regla(
                nombre="R7_cable_penalizacion_clima",
                condiciones=[
                    "linea IN ['J', 'K', 'H', 'L']",
                    "condiciones_climaticas == 'lluvia'"
                ],
                conclusion="agregar_penalizacion_clima(5_minutos)",
                prioridad=2
            ),
            Regla(
                nombre="R8_ruta_optima",
                condiciones=[
                    "ruta_encontrada",
                    "tiempo_total == minimo"
                ],
                conclusion="ruta_es_optima",
                prioridad=3
            ),
        ]

    # --------------------------------------------------------
    # Construcción del grafo para búsqueda
    # --------------------------------------------------------
    def _construir_grafo(self):
        """Construye el grafo de adyacencia a partir de las conexiones."""
        self.grafo = {}
        for estacion in self.estaciones:
            self.grafo[estacion] = []

        for conexion in self.conexiones:
            self.grafo[conexion.origen].append(
                (conexion.destino, conexion.tiempo_minutos, conexion.linea)
            )

    # --------------------------------------------------------
    # Motor de inferencia: evaluación de reglas
    # --------------------------------------------------------
    def evaluar_reglas(self, origen: str, destino: str) -> List[str]:
        """
        Motor de inferencia que evalúa las reglas aplicables.
        Retorna las conclusiones inferidas.
        """
        conclusiones = []
        est_origen = self.estaciones.get(origen)
        est_destino = self.estaciones.get(destino)

        if not est_origen or not est_destino:
            return ["error: estación no encontrada"]

        # R1: Misma línea
        lineas_origen = set(est_origen.lineas_conectadas)
        lineas_destino = set(est_destino.lineas_conectadas)
        if lineas_origen & lineas_destino:
            conclusiones.append("R1: Ruta directa posible (misma línea)")
        else:
            # R2/R3: Verificar transbordos
            transbordo_comun = self._encontrar_transbordo_comun(origen, destino)
            if transbordo_comun:
                conclusiones.append(
                    f"R2: Transbordo simple posible en: {', '.join(transbordo_comun)}"
                )
            else:
                conclusiones.append("R3: Se requieren múltiples transbordos")

        # R5: Penalización por transbordo
        if not (lineas_origen & lineas_destino):
            conclusiones.append("R5: Aplicar penalización por transbordo (+3 min)")

        # R6: Terminal
        if est_origen.es_terminal or est_destino.es_terminal:
            conclusiones.append("R6: Estación terminal - mayor tiempo de espera")

        return conclusiones

    def _encontrar_transbordo_comun(self, origen: str, destino: str) -> List[str]:
        """Encuentra estaciones de transbordo que conectan las líneas de origen y destino."""
        est_origen = self.estaciones[origen]
        est_destino = self.estaciones[destino]
        transbordos = []

        for nombre, est in self.estaciones.items():
            if est.es_transbordo:
                lineas_est = set(est.lineas_conectadas)
                if (lineas_est & set(est_origen.lineas_conectadas) and
                    lineas_est & set(est_destino.lineas_conectadas)):
                    transbordos.append(nombre)

        return transbordos

    # --------------------------------------------------------
    # Función heurística para A*
    # --------------------------------------------------------
    def heuristica(self, estacion_actual: str, estacion_destino: str) -> float:
        """
        Función heurística h(n): distancia euclidiana entre coordenadas.
        Basado en: Benítez (2014), Cap. 9 - Técnicas basadas en búsquedas heurísticas.
        
        Se convierte a tiempo estimado asumiendo velocidad promedio del metro.
        """
        coord_actual = self.estaciones[estacion_actual].coordenadas
        coord_destino = self.estaciones[estacion_destino].coordenadas

        # Distancia euclidiana en grados, convertida a km aprox.
        dlat = coord_actual[0] - coord_destino[0]
        dlon = coord_actual[1] - coord_destino[1]
        distancia_grados = math.sqrt(dlat**2 + dlon**2)
        distancia_km = distancia_grados * 111  # aprox 111 km por grado

        # Convertir a tiempo: velocidad promedio metro ~35 km/h
        tiempo_estimado = (distancia_km / 35) * 60  # en minutos
        return tiempo_estimado

    def obtener_vecinos(self, estacion: str) -> List[Tuple[str, float, str]]:
        """Retorna los vecinos de una estación: [(nombre, tiempo, línea), ...]"""
        return self.grafo.get(estacion, [])

    def listar_estaciones(self) -> List[str]:
        """Retorna lista ordenada de todas las estaciones."""
        return sorted(self.estaciones.keys())

    def obtener_info_estacion(self, nombre: str) -> Optional[Estacion]:
        """Retorna información de una estación."""
        return self.estaciones.get(nombre)

    def __repr__(self):
        return (
            f"BaseConocimiento(estaciones={len(self.estaciones)}, "
            f"conexiones={len(self.conexiones)//2}, "
            f"reglas={len(self.reglas)})"
        )
