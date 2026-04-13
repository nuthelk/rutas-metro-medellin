"""
MOTOR DE BÚSQUEDA HEURÍSTICA - Algoritmo A*
=============================================
Implementación del algoritmo A* para encontrar la ruta óptima
en el sistema de transporte masivo de Medellín.

Referencia:
- Benítez, R. (2014). Inteligencia artificial avanzada. Cap. 9.
  Técnicas basadas en búsquedas heurísticas.

El algoritmo A* combina:
- g(n): costo real acumulado desde el origen hasta n
- h(n): heurística (estimación del costo desde n hasta el destino)
- f(n) = g(n) + h(n): función de evaluación total
"""

import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from base_conocimiento import BaseConocimiento


PENALIZACION_TRANSBORDO = 3.0  # minutos (Regla R5)
PENALIZACION_CABLE_CLIMA = 5.0  # minutos (Regla R7)


@dataclass
class NodoBusqueda:
    """Nodo en el espacio de búsqueda A*."""
    estacion: str
    g: float           # costo acumulado real
    h: float           # heurística estimada
    padre: Optional['NodoBusqueda'] = None
    linea_llegada: str = ""  # línea por la que se llegó a este nodo

    @property
    def f(self) -> float:
        """f(n) = g(n) + h(n)"""
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f


@dataclass
class ResultadoBusqueda:
    """Resultado completo de la búsqueda A*."""
    ruta: List[str]
    lineas: List[str]
    tiempo_total: float
    distancia_total: float
    num_transbordos: int
    nodos_explorados: int
    nodos_generados: int
    pasos_detallados: List[Dict]
    reglas_aplicadas: List[str]


class BuscadorRutas:
    """
    Motor de búsqueda A* para el sistema de transporte.
    
    Implementa búsqueda heurística informada (A*) donde:
    - El grafo es el mapa del sistema de transporte
    - Los pesos son tiempos de viaje entre estaciones
    - La heurística es la distancia euclidiana convertida a tiempo
    - Se aplican penalizaciones según las reglas de la base de conocimiento
    """

    def __init__(self, base: BaseConocimiento, clima: str = "normal"):
        self.base = base
        self.clima = clima

    def buscar_ruta(self, origen: str, destino: str) -> Optional[ResultadoBusqueda]:
        """
        Ejecuta el algoritmo A* para encontrar la ruta óptima.
        
        Parámetros:
            origen: nombre de la estación de origen
            destino: nombre de la estación de destino
            
        Retorna:
            ResultadoBusqueda con la ruta óptima, o None si no existe ruta.
        """
        # Validar estaciones
        if origen not in self.base.estaciones:
            print(f"  [ERROR] Estación '{origen}' no encontrada.")
            return None
        if destino not in self.base.estaciones:
            print(f"  [ERROR] Estación '{destino}' no encontrada.")
            return None
        if origen == destino:
            print("  [INFO] Origen y destino son la misma estación.")
            return ResultadoBusqueda(
                ruta=[origen], lineas=[], tiempo_total=0,
                distancia_total=0, num_transbordos=0,
                nodos_explorados=0, nodos_generados=0,
                pasos_detallados=[], reglas_aplicadas=[]
            )

        # Evaluar reglas antes de la búsqueda
        reglas = self.base.evaluar_reglas(origen, destino)

        # Inicialización A*
        nodo_inicio = NodoBusqueda(
            estacion=origen,
            g=0.0,
            h=self.base.heuristica(origen, destino)
        )

        # Cola de prioridad (min-heap)
        abiertos: List[NodoBusqueda] = []
        heapq.heappush(abiertos, nodo_inicio)

        # Conjunto de cerrados y mejor g conocido
        cerrados = set()
        mejor_g: Dict[str, float] = {origen: 0.0}

        nodos_explorados = 0
        nodos_generados = 1
        pasos = []

        while abiertos:
            # Extraer nodo con menor f(n)
            nodo_actual = heapq.heappop(abiertos)

            # Registrar paso
            pasos.append({
                "paso": nodos_explorados + 1,
                "estacion": nodo_actual.estacion,
                "g": round(nodo_actual.g, 2),
                "h": round(nodo_actual.h, 2),
                "f": round(nodo_actual.f, 2),
                "linea": nodo_actual.linea_llegada or "inicio"
            })

            # ¿Llegamos al destino?
            if nodo_actual.estacion == destino:
                return self._reconstruir_resultado(
                    nodo_actual, nodos_explorados, nodos_generados,
                    pasos, reglas
                )

            # Marcar como explorado
            if nodo_actual.estacion in cerrados:
                continue
            cerrados.add(nodo_actual.estacion)
            nodos_explorados += 1

            # Expandir vecinos
            for vecino, tiempo, linea in self.base.obtener_vecinos(nodo_actual.estacion):
                if vecino in cerrados:
                    continue

                # Calcular g(n) para el vecino
                costo = tiempo

                # Regla R5: penalización por transbordo
                if (nodo_actual.linea_llegada and
                    nodo_actual.linea_llegada != linea):
                    costo += PENALIZACION_TRANSBORDO

                # Regla R7: penalización por clima en metrocables
                if (self.clima == "lluvia" and
                    linea in ["J", "K", "H", "L"]):
                    costo += PENALIZACION_CABLE_CLIMA

                nuevo_g = nodo_actual.g + costo

                # ¿Es mejor camino?
                if vecino not in mejor_g or nuevo_g < mejor_g[vecino]:
                    mejor_g[vecino] = nuevo_g
                    h = self.base.heuristica(vecino, destino)

                    nodo_vecino = NodoBusqueda(
                        estacion=vecino,
                        g=nuevo_g,
                        h=h,
                        padre=nodo_actual,
                        linea_llegada=linea
                    )

                    heapq.heappush(abiertos, nodo_vecino)
                    nodos_generados += 1

        # No se encontró ruta
        print(f"  [ERROR] No se encontró ruta de '{origen}' a '{destino}'.")
        return None

    def _reconstruir_resultado(
        self, nodo_final: NodoBusqueda,
        nodos_explorados: int, nodos_generados: int,
        pasos: List[Dict], reglas: List[str]
    ) -> ResultadoBusqueda:
        """Reconstruye la ruta desde el nodo final siguiendo los padres."""
        ruta = []
        lineas = []
        nodo = nodo_final

        while nodo is not None:
            ruta.append(nodo.estacion)
            if nodo.linea_llegada:
                lineas.append(nodo.linea_llegada)
            nodo = nodo.padre

        ruta.reverse()
        lineas.reverse()

        # Calcular transbordos
        num_transbordos = 0
        for i in range(1, len(lineas)):
            if lineas[i] != lineas[i - 1]:
                num_transbordos += 1

        # Calcular distancia total
        distancia_total = 0.0
        for i in range(len(ruta) - 1):
            for con in self.base.conexiones:
                if con.origen == ruta[i] and con.destino == ruta[i + 1]:
                    distancia_total += con.distancia_km
                    break

        return ResultadoBusqueda(
            ruta=ruta,
            lineas=lineas,
            tiempo_total=round(nodo_final.g, 2),
            distancia_total=round(distancia_total, 2),
            num_transbordos=num_transbordos,
            nodos_explorados=nodos_explorados,
            nodos_generados=nodos_generados,
            pasos_detallados=pasos,
            reglas_aplicadas=reglas
        )
