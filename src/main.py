"""
SISTEMA INTELIGENTE DE RUTAS - Metro de Medellín
==================================================
Programa principal con interfaz de línea de comandos.

Proyecto: Sistema inteligente basado en reglas lógicas y búsqueda
heurística A* para encontrar la mejor ruta en el sistema de
transporte masivo de Medellín.

Componentes teóricos (Benítez, 2014):
- Cap. 2: Lógica y representación del conocimiento
- Cap. 3: Sistemas basados en reglas
- Cap. 9: Técnicas basadas en búsquedas heurísticas
"""

import sys
import os

# Agregar directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_conocimiento import BaseConocimiento
from motor_busqueda import BuscadorRutas


def imprimir_banner():
    print("=" * 65)
    print("   SISTEMA INTELIGENTE DE RUTAS")
    print("   Metro de Medellín - Transporte Masivo")
    print("=" * 65)
    print("   Base de conocimiento con reglas lógicas + Búsqueda A*")
    print("   Líneas: A, B, K, J, T-A, H, L")
    print("=" * 65)


def imprimir_estaciones(base: BaseConocimiento):
    """Muestra todas las estaciones agrupadas por línea."""
    lineas = {}
    for nombre, est in base.estaciones.items():
        linea = est.linea
        if linea not in lineas:
            lineas[linea] = []
        lineas[linea].append(nombre)

    nombres_lineas = {
        "A": "Línea A (Niquía - La Estrella)",
        "B": "Línea B (San Antonio - San Javier)",
        "K": "Línea K - Metrocable (Acevedo - Santo Domingo)",
        "J": "Línea J - Metrocable (San Javier - La Aurora)",
        "T-A": "Tranvía T-A (San Antonio - Oriente)",
        "H": "Línea H - Metrocable (Miraflores - 13 de Noviembre)",
        "L": "Línea L - Metrocable (Santo Domingo - Arví)",
    }

    for clave in ["A", "B", "K", "J", "T-A", "H", "L"]:
        if clave in lineas:
            print(f"\n  {nombres_lineas.get(clave, clave)}:")
            estaciones_linea = lineas[clave]
            for i, est in enumerate(estaciones_linea):
                info = base.estaciones[est]
                marcas = []
                if info.es_terminal:
                    marcas.append("TERMINAL")
                if info.es_transbordo:
                    marcas.append(f"TRANSBORDO: {', '.join(info.lineas_conectadas)}")
                marca_str = f" [{', '.join(marcas)}]" if marcas else ""
                print(f"    {i+1:2d}. {est}{marca_str}")


def imprimir_resultado(resultado):
    """Imprime el resultado de la búsqueda de forma detallada."""
    print("\n" + "=" * 65)
    print("   RESULTADO DE LA BÚSQUEDA")
    print("=" * 65)

    # Reglas aplicadas
    print("\n  REGLAS INFERIDAS:")
    for regla in resultado.reglas_aplicadas:
        print(f"    -> {regla}")

    # Ruta encontrada
    print("\n  RUTA ÓPTIMA ENCONTRADA:")
    print(f"    Estaciones: {len(resultado.ruta)}")
    print(f"    Tiempo total: {resultado.tiempo_total} minutos")
    print(f"    Distancia total: {resultado.distancia_total} km")
    print(f"    Transbordos: {resultado.num_transbordos}")

    # Detalle paso a paso
    print("\n  RECORRIDO DETALLADO:")
    linea_actual = ""
    for i, estacion in enumerate(resultado.ruta):
        if i < len(resultado.lineas):
            nueva_linea = resultado.lineas[i]
        else:
            nueva_linea = linea_actual

        if nueva_linea != linea_actual and i > 0:
            print(f"    {'':4s}  --- TRANSBORDO a Línea {nueva_linea} ---")

        prefijo = ">>>" if i == 0 or i == len(resultado.ruta) - 1 else "   "
        linea_display = nueva_linea if i < len(resultado.lineas) else linea_actual
        print(f"    {prefijo} [{linea_display:3s}] {estacion}")
        linea_actual = nueva_linea

    # Estadísticas de búsqueda A*
    print("\n  ESTADÍSTICAS DEL ALGORITMO A*:")
    print(f"    Nodos explorados: {resultado.nodos_explorados}")
    print(f"    Nodos generados:  {resultado.nodos_generados}")
    print(f"    Eficiencia:       {resultado.nodos_explorados}/{resultado.nodos_generados} "
          f"({resultado.nodos_explorados/max(resultado.nodos_generados,1)*100:.1f}%)")

    # Tabla de exploración
    print("\n  TRAZA DEL ALGORITMO A* (primeros 15 pasos):")
    print(f"    {'Paso':>4s} | {'Estación':<25s} | {'g(n)':>6s} | {'h(n)':>6s} | {'f(n)':>6s} | Línea")
    print(f"    {'----':>4s}-+-{'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+------")
    for paso in resultado.pasos_detallados[:15]:
        print(f"    {paso['paso']:4d} | {paso['estacion']:<25s} | {paso['g']:6.2f} | "
              f"{paso['h']:6.2f} | {paso['f']:6.2f} | {paso['linea']}")
    if len(resultado.pasos_detallados) > 15:
        print(f"    ... ({len(resultado.pasos_detallados) - 15} pasos más)")


def modo_interactivo():
    """Ejecuta el sistema en modo interactivo."""
    imprimir_banner()

    print("\n  Cargando base de conocimiento...")
    base = BaseConocimiento()
    print(f"  {base}")
    print("  Base de conocimiento cargada exitosamente.")

    while True:
        print("\n" + "-" * 50)
        print("  MENÚ PRINCIPAL:")
        print("    1. Buscar ruta óptima")
        print("    2. Buscar ruta (con lluvia)")
        print("    3. Listar estaciones")
        print("    4. Información de estación")
        print("    5. Ver reglas del sistema")
        print("    6. Ejecutar pruebas automáticas")
        print("    0. Salir")
        print("-" * 50)

        opcion = input("  Seleccione opción: ").strip()

        if opcion == "0":
            print("\n  ¡Hasta luego!")
            break

        elif opcion == "1" or opcion == "2":
            clima = "lluvia" if opcion == "2" else "normal"
            if clima == "lluvia":
                print("  [CLIMA] Modo lluvia activado (penalización en metrocables)")

            print("\n  Estaciones disponibles (escriba el nombre exacto):")
            imprimir_estaciones(base)

            origen = input("\n  Estación de ORIGEN: ").strip()
            destino = input("  Estación de DESTINO: ").strip()

            buscador = BuscadorRutas(base, clima=clima)
            resultado = buscador.buscar_ruta(origen, destino)

            if resultado:
                imprimir_resultado(resultado)

        elif opcion == "3":
            imprimir_estaciones(base)

        elif opcion == "4":
            nombre = input("  Nombre de la estación: ").strip()
            info = base.obtener_info_estacion(nombre)
            if info:
                print(f"\n  Estación: {info.nombre}")
                print(f"  Línea principal: {info.linea}")
                print(f"  Coordenadas: {info.coordenadas}")
                print(f"  Es terminal: {'Sí' if info.es_terminal else 'No'}")
                print(f"  Es transbordo: {'Sí' if info.es_transbordo else 'No'}")
                print(f"  Líneas conectadas: {', '.join(info.lineas_conectadas)}")
                vecinos = base.obtener_vecinos(nombre)
                print(f"  Conexiones directas:")
                for v, t, l in vecinos:
                    print(f"    -> {v} (Línea {l}, {t} min)")
            else:
                print(f"  Estación '{nombre}' no encontrada.")

        elif opcion == "5":
            print("\n  REGLAS DEL SISTEMA EXPERTO:")
            for regla in base.reglas:
                print(f"\n  {regla.nombre} (prioridad: {regla.prioridad}):")
                print(f"    SI:")
                for cond in regla.condiciones:
                    print(f"      - {cond}")
                print(f"    ENTONCES: {regla.conclusion}")

        elif opcion == "6":
            ejecutar_pruebas_automaticas()

        else:
            print("  Opción no válida.")


def ejecutar_pruebas_automaticas():
    """Ejecuta un conjunto de pruebas automáticas."""
    print("\n" + "=" * 65)
    print("   PRUEBAS AUTOMÁTICAS DEL SISTEMA")
    print("=" * 65)

    base = BaseConocimiento()

    pruebas = [
        ("Itagüí", "Universidad", "normal",
         "Prueba 1: Misma línea (Línea A)"),
        ("Niquía", "San Javier", "normal",
         "Prueba 2: Transbordo en San Antonio (A -> B)"),
        ("La Estrella", "Santo Domingo", "normal",
         "Prueba 3: Ruta larga con transbordo (A -> K)"),
        ("San Javier", "Oriente", "normal",
         "Prueba 4: Múltiples transbordos (B -> T-A)"),
        ("La Aurora", "Arví", "normal",
         "Prueba 5: Cables y múltiples transbordos (J -> K -> L)"),
        ("Poblado", "Santo Domingo", "lluvia",
         "Prueba 6: Ruta con lluvia (penalización cable)"),
        ("Poblado", "Santo Domingo", "normal",
         "Prueba 7: Misma ruta sin lluvia (comparación)"),
    ]

    for origen, destino, clima, descripcion in pruebas:
        print(f"\n{'─' * 65}")
        print(f"  {descripcion}")
        print(f"  {origen} -> {destino} [clima: {clima}]")
        print(f"{'─' * 65}")

        buscador = BuscadorRutas(base, clima=clima)
        resultado = buscador.buscar_ruta(origen, destino)

        if resultado:
            imprimir_resultado(resultado)
        else:
            print("  No se encontró ruta.")

    print(f"\n{'=' * 65}")
    print("  PRUEBAS COMPLETADAS")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--pruebas":
            ejecutar_pruebas_automaticas()
        elif sys.argv[1] == "--ruta" and len(sys.argv) >= 4:
            base = BaseConocimiento()
            clima = sys.argv[4] if len(sys.argv) > 4 else "normal"
            buscador = BuscadorRutas(base, clima=clima)
            resultado = buscador.buscar_ruta(sys.argv[2], sys.argv[3])
            if resultado:
                imprimir_resultado(resultado)
        elif sys.argv[1] == "--help":
            print("Uso:")
            print("  python main.py                    # Modo interactivo")
            print("  python main.py --pruebas          # Ejecutar pruebas")
            print("  python main.py --ruta ORIGEN DEST # Buscar ruta")
            print("  python main.py --ruta ORIG DEST lluvia  # Con lluvia")
        else:
            print("Argumento no reconocido. Use --help para ver opciones.")
    else:
        modo_interactivo()
