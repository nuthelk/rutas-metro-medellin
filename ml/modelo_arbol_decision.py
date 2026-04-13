"""
MODELO DE APRENDIZAJE SUPERVISADO - Árbol de Decisión
======================================================
Modelo de aprendizaje automático basado en árboles de decisión
para predecir la satisfacción de usuarios del Metro de Medellín.

Referencia:
- Palma Méndez, J. T. (2008). Inteligencia artificial: métodos,
  técnicas y aplicaciones. Cap. 17: Árboles y reglas de decisión.

Componentes:
1. Carga y exploración del dataset
2. Preprocesamiento de datos
3. Entrenamiento del árbol de decisión (CART)
4. Evaluación del modelo
5. Extracción de reglas de decisión
6. Visualización del árbol
"""

import os
import csv
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Importar generador de datos
from dataset_viajes import generar_dataset, guardar_csv


def cargar_o_generar_datos(archivo_csv="dataset_viajes_metro.csv"):
    """Carga el dataset desde CSV o lo genera si no existe."""
    if os.path.exists(archivo_csv):
        print(f"  Cargando dataset desde {archivo_csv}...")
        df = pd.read_csv(archivo_csv)
    else:
        print("  Generando dataset sintético...")
        registros = generar_dataset(500)
        guardar_csv(registros, archivo_csv)
        df = pd.read_csv(archivo_csv)
    return df


def explorar_datos(df):
    """Exploración inicial del dataset."""
    print("\n" + "=" * 65)
    print("  1. EXPLORACIÓN DEL DATASET")
    print("=" * 65)

    print(f"\n  Dimensiones: {df.shape[0]} registros x {df.shape[1]} columnas")
    print(f"\n  Columnas del dataset:")
    for col in df.columns:
        print(f"    - {col}: {df[col].dtype} ({df[col].nunique()} valores únicos)")

    print(f"\n  Primeros 5 registros:")
    print(df.head().to_string(index=False))

    print(f"\n  Estadísticas descriptivas (variables numéricas):")
    print(df.describe().round(2).to_string())

    print(f"\n  Distribución de la variable objetivo (satisfaccion_usuario):")
    dist = df['satisfaccion_usuario'].value_counts()
    for clase, conteo in dist.items():
        print(f"    {clase:6s}: {conteo:4d} ({conteo/len(df)*100:5.1f}%)")

    print(f"\n  Valores nulos por columna:")
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        print("    No hay valores nulos en el dataset.")
    else:
        for col, n in nulos.items():
            if n > 0:
                print(f"    {col}: {n}")

    return df


def preprocesar_datos(df):
    """
    Preprocesamiento del dataset para el modelo.

    Convierte variables categóricas a numéricas usando LabelEncoder.
    Selecciona las features relevantes para el modelo.
    """
    print("\n" + "=" * 65)
    print("  2. PREPROCESAMIENTO DE DATOS")
    print("=" * 65)

    # Features seleccionadas para el modelo
    features_numericas = [
        'misma_linea', 'num_estaciones', 'num_transbordos',
        'usa_metrocable', 'origen_es_terminal', 'destino_es_terminal',
        'es_fin_semana', 'tiempo_viaje_min'
    ]

    features_categoricas = ['congestion', 'clima', 'franja_horaria']

    # Codificar variables categóricas
    encoders = {}
    df_procesado = df.copy()

    for col in features_categoricas:
        le = LabelEncoder()
        df_procesado[col + '_cod'] = le.fit_transform(df_procesado[col])
        encoders[col] = le
        print(f"  Codificación de '{col}':")
        for clase, codigo in zip(le.classes_, le.transform(le.classes_)):
            print(f"    {clase} -> {codigo}")

    # Construir matriz de features
    feature_cols = features_numericas + [c + '_cod' for c in features_categoricas]
    X = df_procesado[feature_cols].values
    y = df_procesado['satisfaccion_usuario'].values

    print(f"\n  Features seleccionadas ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"    - {f}")

    print(f"\n  Forma de X: {X.shape}")
    print(f"  Forma de y: {y.shape}")

    return X, y, feature_cols, encoders


def entrenar_modelo(X, y, feature_cols):
    """
    Entrena el modelo de árbol de decisión.

    Usa el algoritmo CART (Classification and Regression Trees)
    implementado en scikit-learn, que corresponde a la familia
    de algoritmos descrita en Palma Méndez (2008), Cap. 17.
    """
    print("\n" + "=" * 65)
    print("  3. ENTRENAMIENTO DEL MODELO")
    print("=" * 65)

    # División en conjuntos de entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n  División del dataset:")
    print(f"    Entrenamiento: {len(X_train)} registros ({len(X_train)/len(X)*100:.0f}%)")
    print(f"    Prueba:        {len(X_test)} registros ({len(X_test)/len(X)*100:.0f}%)")

    # --- Modelo 1: Árbol sin restricciones ---
    print(f"\n  Modelo 1: Árbol de decisión SIN poda")
    arbol_sin_poda = DecisionTreeClassifier(random_state=42)
    arbol_sin_poda.fit(X_train, y_train)

    acc_train_sp = accuracy_score(y_train, arbol_sin_poda.predict(X_train))
    acc_test_sp = accuracy_score(y_test, arbol_sin_poda.predict(X_test))
    print(f"    Profundidad: {arbol_sin_poda.get_depth()}")
    print(f"    Hojas:       {arbol_sin_poda.get_n_leaves()}")
    print(f"    Accuracy entrenamiento: {acc_train_sp:.4f}")
    print(f"    Accuracy prueba:        {acc_test_sp:.4f}")

    # --- Modelo 2: Árbol con poda (prepruning) ---
    print(f"\n  Modelo 2: Árbol de decisión CON poda (max_depth=5, min_samples_leaf=10)")
    arbol_podado = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        min_samples_split=20,
        random_state=42
    )
    arbol_podado.fit(X_train, y_train)

    acc_train_p = accuracy_score(y_train, arbol_podado.predict(X_train))
    acc_test_p = accuracy_score(y_test, arbol_podado.predict(X_test))
    print(f"    Profundidad: {arbol_podado.get_depth()}")
    print(f"    Hojas:       {arbol_podado.get_n_leaves()}")
    print(f"    Accuracy entrenamiento: {acc_train_p:.4f}")
    print(f"    Accuracy prueba:        {acc_test_p:.4f}")

    # Validación cruzada del modelo podado
    cv_scores = cross_val_score(arbol_podado, X, y, cv=5, scoring='accuracy')
    print(f"\n  Validación cruzada (5-fold) del modelo podado:")
    print(f"    Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"    Media:  {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return arbol_sin_poda, arbol_podado, X_train, X_test, y_train, y_test


def evaluar_modelo(arbol, X_test, y_test, nombre="Modelo"):
    """Evaluación detallada del modelo."""
    print(f"\n" + "=" * 65)
    print(f"  4. EVALUACIÓN: {nombre}")
    print("=" * 65)

    y_pred = arbol.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Reporte de clasificación
    print(f"\n  Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["alta", "baja", "media"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=["alta", "baja", "media"])
    print(f"  Matriz de confusión:")
    print(f"               Pred_alta  Pred_baja  Pred_media")
    labels = ["alta", "baja", "media"]
    for i, label in enumerate(labels):
        print(f"    Real_{label:6s}  {cm[i][0]:6d}     {cm[i][1]:6d}      {cm[i][2]:6d}")

    return y_pred, acc


def extraer_reglas(arbol, feature_cols):
    """
    Extrae las reglas de decisión del árbol entrenado.

    Esto conecta con el Cap. 17 de Palma Méndez (2008):
    los árboles de decisión generan reglas SI-ENTONCES que
    pueden integrarse con el sistema basado en reglas existente.
    """
    print(f"\n" + "=" * 65)
    print(f"  5. REGLAS DE DECISIÓN EXTRAÍDAS")
    print("=" * 65)

    # Reglas en formato texto
    reglas_texto = export_text(
        arbol,
        feature_names=feature_cols,
        class_names=["alta", "baja", "media"],
        max_depth=4
    )
    print(f"\n  Árbol de decisión (primeros 4 niveles):")
    for linea in reglas_texto.split('\n'):
        print(f"    {linea}")

    # Importancia de features
    print(f"\n  Importancia de características (Gini):")
    importancias = list(zip(feature_cols, arbol.feature_importances_))
    importancias.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in importancias:
        barra = "█" * int(imp * 40)
        print(f"    {feat:25s} {imp:.4f} {barra}")

    return importancias


def visualizar_arbol(arbol, feature_cols, archivo="arbol_decision.png"):
    """Genera visualización del árbol de decisión."""
    print(f"\n  Generando visualización del árbol...")

    fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    plot_tree(
        arbol,
        feature_names=feature_cols,
        class_names=["alta", "baja", "media"],
        filled=True,
        rounded=True,
        ax=ax,
        max_depth=3,
        fontsize=8,
        proportion=True
    )
    ax.set_title("Árbol de Decisión - Satisfacción de Usuarios del Metro de Medellín",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Árbol guardado en: {archivo}")
    return archivo


def visualizar_importancia(importancias, archivo="importancia_features.png"):
    """Genera gráfico de importancia de características."""
    features = [f for f, _ in importancias]
    valores = [v for _, v in importancias]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(features)), valores, color='#2196F3', edgecolor='#1565C0')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Importancia (Gini)', fontsize=12)
    ax.set_title('Importancia de Características - Árbol de Decisión',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico guardado en: {archivo}")
    return archivo


def visualizar_matriz_confusion(y_test, y_pred, archivo="matriz_confusion.png"):
    """Genera visualización de la matriz de confusión."""
    cm = confusion_matrix(y_test, y_pred, labels=["alta", "baja", "media"])
    labels = ["alta", "baja", "media"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Matriz de Confusión - Modelo Podado', fontsize=13, fontweight='bold')
    fig.colorbar(im)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Matriz guardada en: {archivo}")
    return archivo


def ejecutar_pipeline():
    """Ejecuta el pipeline completo de aprendizaje automático."""
    print("=" * 65)
    print("  MODELO DE APRENDIZAJE SUPERVISADO")
    print("  Árbol de Decisión - Metro de Medellín")
    print("=" * 65)
    print("  Ref: Palma Méndez (2008), Cap. 17: Árboles de decisión")
    print("=" * 65)

    # 1. Cargar datos
    df = cargar_o_generar_datos()

    # 2. Exploración
    df = explorar_datos(df)

    # 3. Preprocesamiento
    X, y, feature_cols, encoders = preprocesar_datos(df)

    # 4. Entrenamiento
    arbol_sp, arbol_p, X_train, X_test, y_train, y_test = entrenar_modelo(
        X, y, feature_cols
    )

    # 5. Evaluación del modelo podado
    y_pred, acc = evaluar_modelo(arbol_p, X_test, y_test, "Árbol con poda")

    # 6. Reglas de decisión
    importancias = extraer_reglas(arbol_p, feature_cols)

    # 7. Visualizaciones
    img_arbol = visualizar_arbol(arbol_p, feature_cols)
    img_importancia = visualizar_importancia(importancias)
    img_confusion = visualizar_matriz_confusion(y_test, y_pred)

    # Resumen final
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    print(f"  Dataset:              500 viajes simulados")
    print(f"  Features:             {len(feature_cols)} características")
    print(f"  Modelo:               Árbol de decisión CART (podado)")
    print(f"  Profundidad máxima:   {arbol_p.get_depth()}")
    print(f"  Accuracy (test):      {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Archivos generados:")
    print(f"    - dataset_viajes_metro.csv")
    print(f"    - {img_arbol}")
    print(f"    - {img_importancia}")
    print(f"    - {img_confusion}")
    print("=" * 65)

    return arbol_p, df, acc


if __name__ == "__main__":
    arbol, df, acc = ejecutar_pipeline()
