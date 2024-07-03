import os
import pandas as pd
import intrasom
import http.server
import socketserver
import json
from intrasom.visualization import PlotFactory
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from flask import Flask
from flask import jsonify,request

# Define the server address and port
host = "localhost"
port = 7777

def procesarJSON(data): #Validar que el dataframe sea válido! O que lo haga dart, una de las dos
    df = pd.DataFrame(data)
    # df.set_index(df.columns[0], inplace=True) #Importante, esto le marca que la primera columna no son datos, sino que es la etiqueta/nombre
    try:
        df = df.astype(float)
    except:
        raise
    return df

def procesarJSON_string(data): #Validar que el dataframe sea válido! O que lo haga dart, una de las dos
    df = pd.DataFrame(data)
    # df.set_index(df.columns[0], inplace=True) #Importante, esto le marca que la primera columna no son datos, sino que es la etiqueta/nombre
    try:
        df = df.astype(str)
    except:
        raise
    return df

def train(data,params):
    fil = int(params["filas"])
    col = int(params["columnas"])  
    fvecindad = params["vecindad"]
    inicializa = params["inicializacion"]
    rough = int(params["rough"])
    if(rough==0):
        rough = None
    finetuning = int(params["finetuning"])
    if(finetuning==0):
        rough = None
    mapsize = (col,fil)
    som_test = intrasom.SOMFactory.build(data,
        #mask=-9999,
        mapsize=mapsize,
        mapshape='planar',
        lattice='hexa',
        normalization='var',
        initialization= inicializa,
        neighborhood=fvecindad,
        training='batch',
        name='Ejemplo',
        component_names=None,
        unit_names = None,
        sample_names=None,
        #missing=True,
        #save_nan_hist = True,
        pred_size=0)
    som_test.train(summary=False,train_rough_len=rough,train_finetune_len=finetuning, previous_epoch = True)
    #EL SUMMARY EN FALSE ES PARA QUE NO GENERE LOS TXT MOLESTOS, PERO OJO QUE EN ALGUN MOMENTO PODRÍAMOS SACER INFO IMPORTANTE DE AHÍ

    return som_test

def tuplas_umat(som_test):
    plot = PlotFactory(som_test)
    # ruta completa para los archivos
    um = plot.build_umatrix(expanded = True)
    umat = plot.build_umatrix(expanded = False)
    # Matriz en la uqe voy a guardar los colores de las neuronas
    matriz_resultante = np.zeros((2 * som_test.mapsize[1], 2 * som_test.mapsize[0]))
    # Cada neurona me da cuatro valores, por lo que por cada una modico cuatro lugares en l amatriz
    for j in range(som_test.mapsize[1]):
        for i in range(som_test.mapsize[0]):
            valor_actual = umat[j][i]
            vecino_derecha = -1
            if(not np.isnan(um[j, i, 0])):
                vecino_derecha = um[j, i, 0]
            vecino_abajo_derecha = -1
            if(not np.isnan(um[j, i, 1])):    
                vecino_abajo_derecha = um[j, i, 1]
            vecino_abajo_izquierda = -1
            if(not np.isnan(um[j, i, 2])):
                vecino_abajo_izquierda = um[j, i, 2]            
            
            # Modificar los valores en la matriz_resultante
            if j == 0 or j % 2 == 0:
                matriz_resultante[2 * j, 2 * i] = valor_actual # Valor actual
                matriz_resultante[2 * j, 2 * i + 1] = vecino_derecha  # Vecino a la derecha
                matriz_resultante[2 * j + 1, 2 * i] = vecino_abajo_derecha  # Vecino abajo derecha
                if i != 0:
                    # Si es la primera columna, el vecino de abajo a la izquierda no se carga porque no existe ni es valor en la matriz
                    matriz_resultante[2 * j + 1, 2 * i - 1] = vecino_abajo_izquierda  # Vecino abajo izquierda
                if i == (som_test.mapsize[0] -1):
                    # si es la ultima columna, agrego un menos uno en el valor de la derecha al vecino de abajo derecha
                    matriz_resultante[2 * j + 1, 2 * i + 1] = -1
            else:
                matriz_resultante[2 * j, 2 * i + 1] = valor_actual# Valor actual
                if i != (som_test.mapsize[0] -1):
                    # Si es la ultima columna, el vecino de la derecha no se carga porque no existe ni existe el valor en la matriz
                    matriz_resultante[2 * j, 2 * i + 2] = vecino_derecha  # Vecino a la derecha
                matriz_resultante[2 * j + 1, 2 * i] = vecino_abajo_izquierda  # Vecino abajo izquierda
                matriz_resultante[2 * j + 1, 2 * i + 1] = vecino_abajo_derecha  # Vecino abajo derecha
                if i == 0:
                    # Si es la primera columna, el vecino izquierda (al lado del actual) es -1, porque corro toda la fila un lugar
                    matriz_resultante[2 * j, 2 * i] = -1

    # Redondeo a dos decimales
    matriz_redondeada = np.round(matriz_resultante, decimals=2)
    # Mostrar la matriz redondeada
    print(matriz_redondeada)
    # Escribo en un archivo el numero de neurona con el valor normalizado del color
    mi_matriz = matriz_redondeada
    # # Crear una lista de tuplas con los pares (BMU, Dist)
    datos_para_csv = [(i + 1, mi_matriz[i, j]) for i in range(mi_matriz.shape[0]) for j in range(mi_matriz.shape[1]) if not np.isnan(mi_matriz[i, j])]
    # Modifica la primera columna con valores secuenciales (1,2,3,4,5...)
    datos_para_csv = [(i + 1, j) for i, (_, j) in enumerate(datos_para_csv)]
    return datos_para_csv

def df_etiquetas(results_dataframe, etiquetas):
    json_data_etiquetas = json.loads(etiquetas)
    data_etiquetas = procesarJSON_string(json_data_etiquetas)
    etiquetas_df = pd.DataFrame(data_etiquetas)
    etiquetas_df.columns = [col.strip() for col in etiquetas_df.columns]

    nuevo_df = pd.DataFrame({
        'Dato': range(len(results_dataframe)),  # Asumiendo que quieres numerar cada fila como 'Dato'
        'BMU': results_dataframe['BMU']
    })

    for column in etiquetas_df.columns:
        nuevo_df[column] = etiquetas_df[column].str.strip()
    
    return nuevo_df

def tuplas_hits(som_test):
    bmus = som_test._bmu[0].astype(int)
    unique, counts = np.unique(bmus, return_counts=True)
    unique = unique.astype(int).tolist()
    counts = counts.astype(int).tolist()
    return dict(zip(unique, counts))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    print("GET")
    return jsonify({"message": "Welcome to my Flask API"})


@app.route('/bmu', methods=['POST'])
def bmu_return():
    print("POST")
    payload = request.get_json(force=True)
    json_data = payload.get("datos")
    json_data = json.loads(json_data)
    data = procesarJSON(json_data)
    etiquetas = payload.get("etiquetas")
    params = payload.get("params")
    resultados_entrenamiento = train(data,params)
    # ARMO RESPUESTA ETIQUETAS
    etiquetas_df = df_etiquetas(resultados_entrenamiento.results_dataframe, etiquetas)
    etiquetas_df = pd.DataFrame.to_json(etiquetas_df)
    etiquetas_df = json.dumps(etiquetas_df)
    
    # ARMO RESPUESTA hits
    resultado_hits = tuplas_hits(resultados_entrenamiento)
    json.dumps(resultado_hits)

    # ARMO RESPUESTA umat
    resultado_umat = tuplas_umat(resultados_entrenamiento)
    json.dumps(resultado_umat, indent = 2)

    # ARMO RESPUESTA codebook
    codebook = resultados_entrenamiento.codebook.matrix
    #Desnormalizacion de codebook
    codebook = resultados_entrenamiento._normalizer.denormalize_by(data,codebook)
    # codebook = np.round(codebook)

    #Medias y dispersiones
    #me, st = resultados_entrenamiento._normalizer._mean_and_standard_dev(data)
  
    # ARMO RESPUESTA BMU
    resultados_entrenamiento = resultados_entrenamiento.neurons_dataframe
    resultados_entrenamiento = pd.DataFrame.to_json(resultados_entrenamiento)
    resultados_entrenamiento = json.dumps(resultados_entrenamiento)
   
    # DEVUELVO INFO
    jsondata = {}
    jsondata['Datos'] = json_data
    jsondata['Neurons'] = resultados_entrenamiento
    jsondata['UMat'] = resultado_umat
    jsondata['Codebook']= codebook.tolist()
    jsondata['Hits'] = resultado_hits
    jsondata['Etiquetas'] = etiquetas_df
    jsondata["Parametros"] = {"filas":params["filas"],"columnas":params["columnas"]} 
    jsondata = json.dumps(jsondata)
    jsondata = jsondata.replace('\\','') #ESTO NO LO PUDE ARREGLAR DE OTRA FORMA. (Funciona OK de todas formas)
    jsondata = jsondata.replace('""','') #El JSON tiene caracteres extraños/malformados, los elimine asi, pero probablemente sea un arraste de error de algo anterior.
    jsondata = json.loads(jsondata)
    response = {"Datos": jsondata['Datos'],
                'Neurons':jsondata['Neurons'],
                'UMat':jsondata['UMat'],
                'Codebook':jsondata['Codebook'],
                'Hits':jsondata['Hits'],
                'Etiquetas':jsondata['Etiquetas'],
                "Parametros":jsondata["Parametros"]}
    print(response)
    return jsonify(response)

print(f"Server running on {host}:{port}")
os.makedirs('Results', exist_ok=True)
if __name__ == '__main__':
    app.run(host="127.0.0.1",port=7777,debug=True)