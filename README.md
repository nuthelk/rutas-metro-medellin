# Sistema Inteligente de Rutas - Metro de Medellín

## Descripción

Sistema inteligente desarrollado en Python que, a partir de una **base de conocimiento escrita en reglas lógicas**, encuentra la **mejor ruta** para moverse entre dos estaciones del sistema de transporte masivo de Medellín, utilizando el algoritmo de búsqueda heurística **A***.

## Fundamentos Teóricos

El proyecto se basa en los siguientes capítulos de:
> Benítez, R. (2014). *Inteligencia artificial avanzada*. Barcelona: Editorial UOC.

| Capítulo | Tema | Aplicación en el proyecto |
|----------|------|---------------------------|
| Cap. 2 | Lógica y representación del conocimiento | Modelado de estaciones, conexiones y propiedades como hechos lógicos |
| Cap. 3 | Sistemas basados en reglas | 8 reglas SI-ENTONCES para inferencia sobre rutas y condiciones |
| Cap. 9 | Técnicas basadas en búsquedas heurísticas | Algoritmo A* con heurística de distancia euclidiana |

## Arquitectura del Sistema

```
src/
├── base_conocimiento.py   # Hechos (estaciones, conexiones) y reglas lógicas
├── motor_busqueda.py      # Algoritmo A* con penalizaciones por reglas
└── main.py                # Interfaz CLI y pruebas automáticas
```

### Componentes:

1. **Base de Conocimiento** (`base_conocimiento.py`):
   - **Hechos**: 48+ estaciones del Metro de Medellín con coordenadas, líneas (A, B, K, J, T-A, H, L), tipo (terminal/transbordo)
   - **Conexiones**: Grafo bidireccional con tiempos y distancias reales aproximadas
   - **Reglas lógicas**: 8 reglas del sistema experto (R1-R8) para inferencia
   - **Función heurística**: Distancia euclidiana entre coordenadas geográficas

2. **Motor de Búsqueda** (`motor_busqueda.py`):
   - Implementación del algoritmo **A*** (búsqueda heurística informada)
   - `f(n) = g(n) + h(n)` donde g es costo real y h es heurística
   - Penalización de 3 min por transbordo (Regla R5)
   - Penalización de 5 min por metrocable en lluvia (Regla R7)

3. **Programa Principal** (`main.py`):
   - Menú interactivo con 6 opciones
   - Modo CLI con argumentos
   - Suite de 7 pruebas automáticas

## Reglas del Sistema Experto

| Regla | Condición | Conclusión |
|-------|-----------|------------|
| R1 | Origen y destino en misma línea | Ruta directa sin transbordo |
| R2 | Diferente línea + transbordo común | Ruta con un transbordo |
| R3 | Diferente línea + sin transbordo común | Múltiples transbordos |
| R4 | Ruta metro y cable disponibles | Preferir metro si diferencia < 5 min |
| R5 | Requiere transbordo | +3 minutos de penalización |
| R6 | Estación es terminal | Mayor tiempo de espera |
| R7 | Metrocable + lluvia | +5 minutos de penalización |
| R8 | Ruta encontrada + tiempo mínimo | La ruta es óptima |

## Requisitos

- **Python 3.8+** (no requiere librerías externas)
- Probado en Python 3.10, 3.11, 3.12

## Instrucciones de Ejecución

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto
```

### 2. Modo interactivo
```bash
python src/main.py
```
Presenta un menú con opciones para:
- Buscar rutas (clima normal o lluvia)
- Listar estaciones
- Ver información de estaciones
- Ver reglas del sistema
- Ejecutar pruebas automáticas

### 3. Modo línea de comandos
```bash
# Buscar una ruta específica
python src/main.py --ruta "Itagüí" "Universidad"

# Buscar ruta con clima de lluvia
python src/main.py --ruta "Poblado" "Santo Domingo" lluvia

# Ejecutar todas las pruebas automáticas
python src/main.py --pruebas

# Ver ayuda
python src/main.py --help
```

## Ejemplo de Salida

```
  RUTA ÓPTIMA ENCONTRADA:
    Estaciones: 8
    Tiempo total: 17.5 minutos
    Distancia total: 6.8 km
    Transbordos: 0

  RECORRIDO DETALLADO:
    >>> [A  ] Itagüí
        [A  ] Envigado
        [A  ] Ayurá
        [A  ] Aguacatala
        [A  ] Poblado
        [A  ] Industriales
        [A  ] Exposiciones
    >>> [A  ] Universidad

  ESTADÍSTICAS DEL ALGORITMO A*:
    Nodos explorados: 8
    Nodos generados:  12
    Eficiencia:       66.7%
```

## Pruebas

El sistema incluye 7 pruebas automáticas que cubren:

1. **Ruta en misma línea** (Itagüí → Universidad)
2. **Transbordo simple** (Niquía → San Javier)
3. **Ruta larga con transbordo** (La Estrella → Santo Domingo)
4. **Múltiples transbordos** (San Javier → Oriente)
5. **Cables con múltiples transbordos** (La Aurora → Arví)
6. **Ruta con lluvia** (Poblado → Santo Domingo)
7. **Misma ruta sin lluvia** (comparación con prueba 6)

Ejecutar: `python src/main.py --pruebas`

## Líneas del Sistema Modeladas

- **Línea A**: Niquía ↔ La Estrella (Metro, 21 estaciones)
- **Línea B**: San Antonio ↔ San Javier (Metro, 7 estaciones)
- **Línea K**: Acevedo ↔ Santo Domingo (Metrocable, 4 estaciones)
- **Línea J**: San Javier ↔ La Aurora (Metrocable, 4 estaciones)
- **Línea T-A**: San Antonio ↔ Oriente (Tranvía, 9 estaciones)
- **Línea H**: Miraflores ↔ 13 de Noviembre (Metrocable, 2 estaciones)
- **Línea L**: Santo Domingo ↔ Arví (Metrocable, 2 estaciones)

## Autores

[Agregar nombres de los integrantes del equipo]

## Referencias

- Benítez, R. (2014). *Inteligencia artificial avanzada*. Barcelona: Editorial UOC.
  - Capítulo 2: Lógica y representación del conocimiento
  - Capítulo 3: Sistemas basados en reglas
  - Capítulo 9: Técnicas basadas en búsquedas heurísticas
