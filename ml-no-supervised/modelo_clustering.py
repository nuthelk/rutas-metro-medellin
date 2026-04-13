"""
MODELO DE APRENDIZAJE NO SUPERVISADO - Clustering
===================================================
Modelo de agrupamiento (clustering) para identificar perfiles
de estaciones del Metro de Medellin sin etiquetas previas.

Referencia:
- Palma Mendez, J. T. (2008). Inteligencia artificial: metodos,
  tecnicas y aplicaciones. Cap. 16: Tecnicas de agrupamiento.

Algoritmos implementados:
1. K-Means: agrupamiento por particiones
2. Clustering Jerarquico: agrupamiento aglomerativo
3. Metodo del Codo + Silhouette para seleccionar K optimo

A diferencia del aprendizaje supervisado (Act. 3), aqui NO hay
variable objetivo. El algoritmo descubre los grupos por si solo.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_estaciones import generar_dataset, guardar_csv


# ============================================================
# 1. CARGA Y EXPLORACION DE DATOS
# ============================================================

def cargar_datos(archivo_csv="dataset_estaciones_metro.csv"):
    """Carga o genera el dataset de estaciones."""
    if os.path.exists(archivo_csv):
        print(f"  Cargando dataset desde {archivo_csv}...")
        df = pd.read_csv(archivo_csv)
    else:
        print("  Generando dataset...")
        registros = generar_dataset()
        guardar_csv(registros, archivo_csv)
        df = pd.read_csv(archivo_csv)
    return df


def explorar_datos(df):
    """Exploracion inicial del dataset."""
    print("\n" + "=" * 65)
    print("  1. EXPLORACION DEL DATASET")
    print("=" * 65)

    print(f"\n  Dimensiones: {df.shape[0]} estaciones x {df.shape[1]} variables")

    print(f"\n  Variables del dataset:")
    for col in df.columns:
        print(f"    - {col}: {df[col].dtype}")

    print(f"\n  Estadisticas descriptivas:")
    numericas = df.select_dtypes(include=[np.number])
    print(numericas.describe().round(2).to_string())

    print(f"\n  NOTA IMPORTANTE: Este dataset NO tiene variable objetivo.")
    print(f"  El algoritmo de clustering descubrira los grupos por si solo.")

    return df


# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================

def preprocesar(df):
    """
    Preprocesamiento para clustering.
    - Selecciona solo variables numericas
    - Estandariza con StandardScaler (media=0, desv=1)
    
    La estandarizacion es fundamental en clustering porque
    K-Means usa distancia euclidiana: sin estandarizar, las
    variables con rangos grandes dominarian el agrupamiento.
    """
    print("\n" + "=" * 65)
    print("  2. PREPROCESAMIENTO")
    print("=" * 65)

    # Variables numericas para clustering
    features = [
        'pasajeros_dia', 'proporcion_hora_pico', 'num_conexiones',
        'tiempo_espera_min', 'indice_congestion', 'distancia_centro_km',
        'frecuencia_servicio', 'incidentes_mes', 'satisfaccion_promedio',
        'es_metrocable'
    ]

    X = df[features].values
    nombres = df['estacion'].values

    print(f"\n  Features seleccionadas ({len(features)}):")
    for f in features:
        print(f"    - {f}")

    # Estandarizacion
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Estandarizacion aplicada (StandardScaler):")
    print(f"    Antes - Media pasajeros_dia: {X[:, 0].mean():.0f}, Desv: {X[:, 0].std():.0f}")
    print(f"    Despues - Media: {X_scaled[:, 0].mean():.4f}, Desv: {X_scaled[:, 0].std():.4f}")

    return X_scaled, features, nombres, scaler


# ============================================================
# 3. SELECCION DE K OPTIMO
# ============================================================

def encontrar_k_optimo(X_scaled, max_k=10):
    """
    Determina el numero optimo de clusters usando:
    1. Metodo del Codo (inercia/distorsion)
    2. Coeficiente de Silhouette

    Palma Mendez (2008), Cap. 16: la seleccion de K es un
    paso critico en algoritmos de particion como K-Means.
    """
    print("\n" + "=" * 65)
    print("  3. SELECCION DE K OPTIMO")
    print("=" * 65)

    rango_k = range(2, max_k + 1)
    inercias = []
    silhouettes = []

    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"    K={k}: Inercia={km.inertia_:.1f}, Silhouette={sil:.4f}")

    # Mejor K por silhouette
    mejor_k = list(rango_k)[np.argmax(silhouettes)]
    print(f"\n  Mejor K segun Silhouette: {mejor_k} (score={max(silhouettes):.4f})")

    # Grafico del codo + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Codo
    ax1.plot(list(rango_k), inercias, 'o-', color='#1565C0', linewidth=2, markersize=6)
    ax1.axvline(x=mejor_k, color='#D85A30', linestyle='--', linewidth=1.5, label=f'K={mejor_k}')
    ax1.set_xlabel('Numero de clusters (K)', fontsize=12)
    ax1.set_ylabel('Inercia (distorsion)', fontsize=12)
    ax1.set_title('Metodo del Codo', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette
    ax2.plot(list(rango_k), silhouettes, 's-', color='#2E7D32', linewidth=2, markersize=6)
    ax2.axvline(x=mejor_k, color='#D85A30', linestyle='--', linewidth=1.5, label=f'K={mejor_k}')
    ax2.set_xlabel('Numero de clusters (K)', fontsize=12)
    ax2.set_ylabel('Coeficiente Silhouette', fontsize=12)
    ax2.set_title('Analisis Silhouette', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metodo_codo_silhouette.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafico guardado: metodo_codo_silhouette.png")

    return mejor_k


# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================

def aplicar_kmeans(X_scaled, k, nombres, df, features):
    """
    Aplica K-Means clustering.

    K-Means (Palma Mendez, 2008, Cap. 16):
    1. Inicializa K centroides aleatoriamente
    2. Asigna cada punto al centroide mas cercano
    3. Recalcula centroides como media del cluster
    4. Repite hasta convergencia
    """
    print("\n" + "=" * 65)
    print(f"  4. K-MEANS CLUSTERING (K={k})")
    print("=" * 65)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    etiquetas = kmeans.fit_predict(X_scaled)

    # Silhouette score
    sil_score = silhouette_score(X_scaled, etiquetas)
    print(f"\n  Silhouette Score global: {sil_score:.4f}")
    print(f"  Iteraciones hasta convergencia: {kmeans.n_iter_}")
    print(f"  Inercia final: {kmeans.inertia_:.1f}")

    # Analisis por cluster
    df_resultado = df.copy()
    df_resultado['cluster'] = etiquetas

    print(f"\n  PERFILES DE CLUSTERS:")
    for c in range(k):
        mask = df_resultado['cluster'] == c
        grupo = df_resultado[mask]
        n = len(grupo)
        print(f"\n  --- Cluster {c} ({n} estaciones) ---")
        print(f"  Estaciones: {', '.join(grupo['estacion'].values)}")
        print(f"  Tipos: {dict(grupo['tipo_estacion'].value_counts())}")
        print(f"  Zonas: {dict(grupo['zona'].value_counts())}")
        print(f"  Metricas promedio:")
        for feat in ['pasajeros_dia', 'indice_congestion', 'tiempo_espera_min',
                      'distancia_centro_km', 'frecuencia_servicio']:
            print(f"    {feat:25s}: {grupo[feat].mean():.1f}")

    return etiquetas, kmeans, df_resultado


# ============================================================
# 5. CLUSTERING JERARQUICO
# ============================================================

def aplicar_jerarquico(X_scaled, k, nombres):
    """
    Aplica clustering jerarquico aglomerativo.

    Metodo aglomerativo (Palma Mendez, 2008, Cap. 16):
    1. Cada punto empieza como su propio cluster
    2. En cada paso, fusiona los dos clusters mas cercanos
    3. Usa criterio de enlace (Ward = minimizar varianza)
    4. Continua hasta tener K clusters
    """
    print("\n" + "=" * 65)
    print(f"  5. CLUSTERING JERARQUICO (K={k})")
    print("=" * 65)

    # Dendrograma
    Z = linkage(X_scaled, method='ward')

    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        Z,
        labels=nombres,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=Z[-(k-1), 2],
        above_threshold_color='gray'
    )
    ax.set_title('Dendrograma - Clustering Jerarquico (Ward)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estaciones', fontsize=12)
    ax.set_ylabel('Distancia (Ward)', fontsize=12)
    ax.axhline(y=Z[-(k-1), 2], color='red', linestyle='--', linewidth=1.5,
               label=f'Corte en K={k}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('dendrograma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Dendrograma guardado: dendrograma.png")

    # Modelo aglomerativo
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    etiquetas_hier = agg.fit_predict(X_scaled)

    sil_hier = silhouette_score(X_scaled, etiquetas_hier)
    print(f"  Silhouette Score (jerarquico): {sil_hier:.4f}")

    # Comparar con K-Means
    print(f"\n  Distribucion de clusters jerarquico:")
    for c in range(k):
        ests = nombres[etiquetas_hier == c]
        print(f"    Cluster {c}: {len(ests)} estaciones")

    return etiquetas_hier


# ============================================================
# 6. VISUALIZACIONES
# ============================================================

def visualizar_clusters(X_scaled, etiquetas, nombres, k, features):
    """Genera visualizaciones de los clusters."""
    print("\n  Generando visualizaciones...")

    from sklearn.decomposition import PCA

    # Reduccion a 2D con PCA para visualizacion
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    print(f"  Varianza explicada por PCA (2 componentes): {sum(pca.explained_variance_ratio_)*100:.1f}%")

    # Grafico de clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    colores = ['#1565C0', '#D85A30', '#2E7D32', '#7B1FA2', '#C62828']
    marcadores = ['o', 's', '^', 'D', 'v']

    for c in range(k):
        mask = etiquetas == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colores[c % len(colores)],
                   marker=marcadores[c % len(marcadores)],
                   s=100, label=f'Cluster {c}',
                   edgecolors='white', linewidth=0.5, alpha=0.85)

    # Etiquetas de estaciones
    for i, nombre in enumerate(nombres):
        ax.annotate(nombre, (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')

    ax.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                  fontsize=12)
    ax.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                  fontsize=12)
    ax.set_title('Clusters de Estaciones - K-Means (proyeccion PCA)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('clusters_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafico de clusters guardado: clusters_pca.png")


def visualizar_radar(df_resultado, k, features_radar=None):
    """Genera grafico radar comparativo de perfiles de clusters."""
    if features_radar is None:
        features_radar = ['pasajeros_dia', 'proporcion_hora_pico', 'indice_congestion',
                          'tiempo_espera_min', 'distancia_centro_km', 'frecuencia_servicio']

    # Normalizar al rango [0, 1] para el radar
    df_norm = df_resultado.copy()
    for f in features_radar:
        min_val = df_norm[f].min()
        max_val = df_norm[f].max()
        if max_val > min_val:
            df_norm[f] = (df_norm[f] - min_val) / (max_val - min_val)

    angles = np.linspace(0, 2 * np.pi, len(features_radar), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colores = ['#1565C0', '#D85A30', '#2E7D32', '#7B1FA2', '#C62828']

    for c in range(k):
        valores = df_norm[df_norm['cluster'] == c][features_radar].mean().values.tolist()
        valores += valores[:1]
        ax.plot(angles, valores, 'o-', color=colores[c % len(colores)],
                linewidth=2, label=f'Cluster {c}', markersize=5)
        ax.fill(angles, valores, color=colores[c % len(colores)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    labels_cortos = [f.replace('_', '\n') for f in features_radar]
    ax.set_xticklabels(labels_cortos, fontsize=9)
    ax.set_title('Perfil Comparativo de Clusters', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('perfil_radar_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafico radar guardado: perfil_radar_clusters.png")


# ============================================================
# 7. INTERPRETACION DE CLUSTERS
# ============================================================

def interpretar_clusters(df_resultado, k):
    """
    Interpreta y nombra cada cluster segun sus caracteristicas.
    Esta es la parte clave del aprendizaje no supervisado:
    el algoritmo agrupa, el humano interpreta.
    """
    print("\n" + "=" * 65)
    print("  7. INTERPRETACION DE CLUSTERS")
    print("=" * 65)

    for c in range(k):
        grupo = df_resultado[df_resultado['cluster'] == c]
        n = len(grupo)

        prom_pasajeros = grupo['pasajeros_dia'].mean()
        prom_congestion = grupo['indice_congestion'].mean()
        prom_espera = grupo['tiempo_espera_min'].mean()
        prom_dist = grupo['distancia_centro_km'].mean()
        prom_cable = grupo['es_metrocable'].mean()

        # Asignar nombre descriptivo
        if prom_cable > 0.5:
            perfil = "ESTACIONES DE METROCABLE"
            descripcion = ("Estaciones de lineas aereas (K, J, H, L). "
                          "Menor frecuencia, mayor tiempo de espera, "
                          "alejadas del centro urbano.")
        elif prom_congestion > 6.5 and prom_pasajeros > 15000:
            perfil = "HUBS DE ALTA DEMANDA"
            descripcion = ("Estaciones centrales y de transbordo con "
                          "alto flujo de pasajeros, alta congestion "
                          "y multiples conexiones.")
        elif prom_dist > 6:
            perfil = "ESTACIONES PERIFERICAS"
            descripcion = ("Estaciones alejadas del centro, menor "
                          "flujo de pasajeros, baja congestion, "
                          "zonas residenciales.")
        else:
            perfil = "ESTACIONES DE PASO URBANAS"
            descripcion = ("Estaciones regulares de zona urbana, "
                          "flujo moderado, congestion media, "
                          "buena frecuencia de servicio.")

        print(f"\n  Cluster {c}: {perfil}")
        print(f"  {descripcion}")
        print(f"  Cantidad: {n} estaciones")
        print(f"  Pasajeros/dia promedio: {prom_pasajeros:,.0f}")
        print(f"  Congestion promedio: {prom_congestion:.1f}/10")
        print(f"  Tiempo espera promedio: {prom_espera:.1f} min")
        print(f"  Distancia al centro promedio: {prom_dist:.1f} km")
        print(f"  Estaciones: {', '.join(grupo['estacion'].values)}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def ejecutar_pipeline():
    """Ejecuta el pipeline completo de clustering."""
    print("=" * 65)
    print("  MODELO DE APRENDIZAJE NO SUPERVISADO")
    print("  Clustering - Estaciones del Metro de Medellin")
    print("=" * 65)
    print("  Ref: Palma Mendez (2008), Cap. 16: Tecnicas de agrupamiento")
    print("=" * 65)

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Explorar
    df = explorar_datos(df)

    # 3. Preprocesar
    X_scaled, features, nombres, scaler = preprocesar(df)

    # 4. Encontrar K optimo
    mejor_k = encontrar_k_optimo(X_scaled)

    # 5. K-Means
    etiquetas, kmeans, df_resultado = aplicar_kmeans(
        X_scaled, mejor_k, nombres, df, features
    )

    # 6. Clustering Jerarquico
    etiquetas_hier = aplicar_jerarquico(X_scaled, mejor_k, nombres)

    # 7. Visualizaciones
    visualizar_clusters(X_scaled, etiquetas, nombres, mejor_k, features)
    visualizar_radar(df_resultado, mejor_k)

    # 8. Interpretacion
    interpretar_clusters(df_resultado, mejor_k)

    # Comparacion K-Means vs Jerarquico
    print("\n" + "=" * 65)
    print("  COMPARACION DE ALGORITMOS")
    print("=" * 65)
    sil_km = silhouette_score(X_scaled, etiquetas)
    sil_hi = silhouette_score(X_scaled, etiquetas_hier)
    print(f"  K-Means    - Silhouette: {sil_km:.4f}")
    print(f"  Jerarquico - Silhouette: {sil_hi:.4f}")
    mejor = "K-Means" if sil_km >= sil_hi else "Jerarquico"
    print(f"  Mejor algoritmo: {mejor}")

    # Resumen
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    print(f"  Dataset:          {len(df)} estaciones, {len(features)} variables")
    print(f"  K optimo:         {mejor_k} clusters")
    print(f"  Mejor algoritmo:  {mejor}")
    print(f"  Silhouette:       {max(sil_km, sil_hi):.4f}")
    print(f"  Archivos generados:")
    print(f"    - dataset_estaciones_metro.csv")
    print(f"    - metodo_codo_silhouette.png")
    print(f"    - dendrograma.png")
    print(f"    - clusters_pca.png")
    print(f"    - perfil_radar_clusters.png")
    print("=" * 65)

    return df_resultado, mejor_k


if __name__ == "__main__":
    df_resultado, k = ejecutar_pipeline()
