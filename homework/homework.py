# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline as build_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

def cargar_datos(nombre_archivo):
    ruta = Path("files/input")
    return pd.read_csv(ruta / nombre_archivo, compression="zip")

def depurar_datos(data: pd.DataFrame):
    data = data.copy()
    data = data.rename(columns={"default payment next month": "default"})
    data = data.drop(columns="ID")
    data = data.dropna()
    data = data[data["MARRIAGE"] != 0]
    data = data[data["EDUCATION"] != 0]

    data["EDUCATION"] = data["EDUCATION"].map(lambda x: 4 if x > 4 else x)
    return data

def separar_variables(data: pd.DataFrame):
    X = data.drop(columns="default").copy()
    y = data["default"].copy()
    return X, y

def dividir_datos(data: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    X, y = separar_variables(data)
    return train_test_split(X, y, random_state=0)

def construir_pipeline():
    transformador = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ['SEX', 'EDUCATION', 'MARRIAGE'])],
        remainder=StandardScaler()
    )

    modelo = build_pipeline(
        transformador,
        PCA(),
        SelectKBest(k=12),
        SVC(gamma=0.1)
    )

    return modelo

def definir_grid_search(modelo, parametros, cv=10):
    return GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

def guardar_modelo(modelo):
    import pickle
    import gzip

    directorio = Path("files/models")
    directorio.mkdir(exist_ok=True)

    with gzip.open(directorio / "model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo, archivo)

def guardar_metricas(lista_metricas):
    import json

    ruta_salida = Path("files/output")
    ruta_salida.mkdir(exist_ok=True)

    with open(ruta_salida / "metrics.json", "w") as archivo:
        archivo.writelines([json.dumps(m) + "\n" for m in lista_metricas])

def entrenar_modelo(X_train, y_train):
    modelo = construir_pipeline()
    grid = definir_grid_search(modelo, {
        "pca__n_components": [20, 21],
    })

    grid.fit(X_train, y_train)
    guardar_modelo(grid)
    return grid

def calcular_metricas(y_real, y_estimado):
    from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

    return (
        precision_score(y_real, y_estimado),
        balanced_accuracy_score(y_real, y_estimado),
        recall_score(y_real, y_estimado),
        f1_score(y_real, y_estimado)
    )

def matriz_confusion(nombre_set, y_real, y_estimado):
    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(y_real, y_estimado)
    return {
        "type": "cm_matrix",
        "dataset": nombre_set,
        "true_0": {
            "predicted_0": int(matriz[0][0]),
            "predicted_1": int(matriz[0][1])
        },
        "true_1": {
            "predicted_0": int(matriz[1][0]),
            "predicted_1": int(matriz[1][1])
        }
    }

def ejecutar_proceso():
    datos_entrenamiento = depurar_datos(cargar_datos("train_data.csv.zip"))
    datos_prueba = depurar_datos(cargar_datos("test_data.csv.zip"))

    X_train, y_train = separar_variables(datos_entrenamiento)
    X_test, y_test = separar_variables(datos_prueba)

    modelo = entrenar_modelo(X_train, y_train)

    resultados = []
    for nombre, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        predicciones = modelo.predict(X)
        precision, bal_acc, sensibilidad, f1 = calcular_metricas(y, predicciones)
        resultados.append({
            "type": "metrics",
            "dataset": nombre,
            "precision": precision,
            "balanced_accuracy": bal_acc,
            "recall": sensibilidad,
            "f1_score": f1
        })

    matrices = [
        matriz_confusion(nombre, y, modelo.predict(X))
        for nombre, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]
    ]

    guardar_metricas(resultados + matrices)

ejecutar_proceso()