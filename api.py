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
from flask import jsonify,request,Response
from flask_cors import CORS

# Define the server address and port
host = "localhost"
port = 7777

def normalizar(datos,datosNormalizar):
    normalizer = intrasom.object_functions.NormalizerFactory.build("var")
    return normalizer.normalize_by(datos,datosNormalizar)
#agregar datos a esta funcion
def find_bmus(datos,som_codebook, input_data_batch):
    datos = np.array(datos)
    som_codebook = np.array(som_codebook)
    input_data_batch = np.array(input_data_batch)
    input_data_batch = normalizar(datos,input_data_batch)
    som_codebook = normalizar(datos,som_codebook)
    # Calculate the Euclidean distance between each input data point and each neuron in the SOM
    distances = np.linalg.norm(som_codebook[:, np.newaxis] - input_data_batch, axis=2)
    # Find the index of the neuron with the minimum distance for each input data point
    bmu_indices = np.argmin(distances, axis=0)
    bmu_indices += 1
    return bmu_indices

def kmeans(datos,codebook,fil,col, k=3, init = "k-means++", n_init=5, max_iter=200):
    datos = np.array(datos)
    codebook = np.array(codebook)
    codebook = normalizar(datos,codebook) # Esto en caso que mandemos el codebook desnormalizado
    kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter).fit(codebook).labels_+1
    return kmeans.reshape(fil,col)

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
    som_test.train(save=False,summary=False,train_rough_len=rough,train_finetune_len=finetuning, previous_epoch = True)
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
CORS(app)
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to my Flask API"})


@app.route('/bmu', methods=['POST'])
def bmu_return():
    try:
        payload = request.get_json(force=True)
        json_data = payload.get("datos")
        json_data = json.loads(json_data)
        data = procesarJSON(json_data)
        etiquetas = payload.get("etiquetas")
        params = payload.get("params")
        resultados_entrenamiento = train(data,params)

        # ARMO RESPUESTA PARAMETROS
        parametros = {"filas":params["filas"],"columnas":params["columnas"],
                      "Lattice":resultados_entrenamiento.lattice,
                      "Neighborhood Function":resultados_entrenamiento.neighborhood.name,
                      "Normalization": resultados_entrenamiento._normalizer.name,
                      "Inicialization": resultados_entrenamiento.initialization,
                      "Rough Training Size": resultados_entrenamiento.train_rough_len,
                      "Rough Training Initial Ratio": resultados_entrenamiento.train_rough_radiusin,
                      "Rough Training Initial Ratio" : resultados_entrenamiento.train_rough_radiusfin,
                      "Fine Training Size": resultados_entrenamiento.train_finetune_len,
                      "Fine Training Initial Ratio": resultados_entrenamiento.train_finetune_radiusin,
                      "Fine Training Final Ratio" : resultados_entrenamiento.train_finetune_radiusfin,
                      }
        
        # ARMO RESPUESTA ERRORES
        errores = {"Topographic": resultados_entrenamiento.topographic_error,
                   "Quantization": resultados_entrenamiento.calculate_quantization_error}
    
        
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
        codebook = np.round(codebook,decimals=2)

        #Medias y dispersiones
        #me, st = resultados_entrenamiento._normalizer._mean_and_standard_dev(data)
    
        # ARMO RESPUESTA BMU
        
        resultados_entrenamiento = resultados_entrenamiento.neurons_dataframe
        resultados_entrenamiento.columns  = [col.replace('B_', '') for col in resultados_entrenamiento.columns]
        resultados_entrenamiento = resultados_entrenamiento.to_json(force_ascii=False)
        resultados_entrenamiento = json.dumps(resultados_entrenamiento, ensure_ascii=False)
     
    
        # DEVUELVO INFO
        jsondata = {}
        jsondata['Datos'] = json_data
        jsondata['Neurons'] = resultados_entrenamiento
        jsondata['UMat'] = resultado_umat
        jsondata['Codebook']= codebook.tolist()
        jsondata['Hits'] = resultado_hits
        jsondata['Etiquetas'] = etiquetas_df
        jsondata["Parametros"] = parametros
        jsondata["Errores"] = errores
        jsondata = json.dumps(jsondata,ensure_ascii=False)
        jsondata = jsondata.replace('\\','') #ESTO NO LO PUDE ARREGLAR DE OTRA FORMA. (Funciona OK de todas formas)
        jsondata = jsondata.replace('""','') #El JSON tiene caracteres extraños/malformados, los elimine asi, pero probablemente sea un arraste de error de algo anterior.
        jsondata = json.loads(jsondata)
      
        return Response(json.dumps(jsondata,ensure_ascii=False), mimetype='application/json; charset=utf-8')
    except Exception as e:
        return jsonify({"error": "Error durante el entrenamiento: "  + str(e)}), 500

@app.route('/clusters', methods=['POST'])
def cluster_return():
    try:
        payload = request.get_json(force=True)
        datos = payload.get("datos")
        codebook = payload.get("codebook")
        params = payload.get("params")
        filas = int(params['filas'])
        columnas = int(params['columnas'])
        cant_clusters = int(params['cantidadClusters'])
        datos = procesarJSON(datos)
        resultado_clustering = kmeans(datos,codebook,filas,columnas,k=cant_clusters)
        resultado_clustering = resultado_clustering.tolist()
        return jsonify(resultado_clustering),200
    except Exception as e:
        return jsonify({"error": "Error durante el clustering: "  + str(e)}), 500


@app.route('/nuevosDatos', methods=['POST'])
def nuevosdatos_return():
    try:
        payload = request.get_json(force=True)
        datos = payload.get("datos")
        nuevosDatos = payload.get("nuevosDatos")
        nuevosDatos = json.loads(nuevosDatos)
        nuevosDatos = [[float(value) for value in entry.values()] for entry in nuevosDatos]
        codebook = payload.get("codebook")
        datos = procesarJSON(datos)
        bmus = find_bmus(datos,codebook,nuevosDatos)
        nuevo_df = pd.DataFrame({
            'Dato': nuevosDatos,  # Asumiendo que quieres numerar cada fila como 'Dato'
            'BMU': bmus
        })

        nuevo_df = pd.DataFrame.to_json(nuevo_df)
        nuevo_df = json.dumps(nuevo_df)
        
        # DEVUELVO INFO
        jsondata = {}
        jsondata['Resultado'] = nuevo_df

        jsondata = json.dumps(jsondata)
        jsondata = jsondata.replace('\\','') 
        jsondata = jsondata.replace('""','') 
        jsondata = json.loads(jsondata)
        response = {"Resultado": jsondata['Resultado']}
        return jsonify(response),200
    except Exception as e:
        return jsonify({"error": "Error durante la llamada a nuevos datos: "  + str(e)}), 500

print(f"Server running on {host}:{port}")
os.makedirs('Results', exist_ok=True)
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=7777,debug=True)
