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
from datetime import datetime

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
    som_test.train(summary=False,train_rough_len=rough,train_finetune_len=finetuning, previous_epoch = True)
    #EL SUMMARY EN FALSE ES PARA QUE NO GENERE LOS TXT MOLESTOS, PERO OJO QUE EN ALGUN MOMENTO PODRÍAMOS SACER INFO IMPORTANTE DE AHÍ

    return som_test

def tuplas_umat(som_test):
    plot = PlotFactory(som_test)
    # ruta completa para los archivos
    um = plot.build_umatrix(expanded = True)
    umat = plot.build_umatrix(expanded = False)
    # Aplana las matrices tridimensionales
    # um_flat = um.flatten()
    # umat_flat = umat.flatten()
    # # Guardar en archivos de texto con las rutas completas
    # np.savetxt(ruta_um, um_flat, fmt='%f', delimiter='\t')
    # np.savetxt(ruta_umat, umat_flat, fmt='%f', delimiter='\t')
    # Funcion para normalizar los valores
    norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
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
            # # Modificar los valores en la matriz_resultante
            # matriz_resultante[2 * j, 2 * i] = valor_actual# Valor actual
            # matriz_resultante[2 * j, 2 * i + 1] = vecino_derecha  # Vecino a la derecha
            # matriz_resultante[2 * j + 1, 2 * i] = vecino_arriba_izquierda  # Vecino arriba izquierda
            # matriz_resultante[2 * j + 1, 2 * i + 1] = vecino_arriba_derecha  # Vecino arriba derecha

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
            
            # bmu = j * som_test.mapsize[0] + i + 1
            # if bmu < 3 or (bmu > 23 and bmu < 27):
            #     print(f"BMU = {bmu}")
            #     print(f"valor_actual = {valor_actual}")
            #     print(f"vecino_derecha = {vecino_derecha}")
            #     print(f"vecino_arriba_derecha = {vecino_arriba_derecha}")
            #     print(f"vecino_arriba_izquierda = {vecino_arriba_izquierda}")
            # if(j==0):
            #     valor_actual = norm(umat[j][i])
            #     vecino_derecha = -1
            #     if(not np.isnan(um[j, i, 0])):
            #         vecino_derecha = norm(um[j, i, 0])
            #     vecino_arriba_derecha = -1
            #     if(not np.isnan(um[j, i, 1])):    
            #         vecino_arriba_derecha = norm(um[j, i, 1])
            #     vecino_arriba_izquierda = -1
            #     if(not np.isnan(um[j, i, 2])):
            #         vecino_arriba_izquierda = norm(um[j, i, 2])              
            #     # Modificar los valores en la matriz_resultante
            #     matriz_resultante[2 * j, 2 * i] = valor_actual  # Valor actual
            #     matriz_resultante[2 * j, 2 * i + 1] = vecino_derecha  # Vecino a la derecha
            #     matriz_resultante[2 * j - 1, 2 * i] = vecino_arriba_derecha  # Vecino arriba izquierda
            #     matriz_resultante[2 * j + 1, 2 * i - 1] = vecino_arriba_izquierda # Vecino arriba derecha

            # if(i==0):
            #     valor_actual = norm(umat[j][i])
            #     vecino_derecha = -1
            #     if(not np.isnan(um[j, i, 0])):
            #         vecino_derecha = norm(um[j, i, 0])
            #     vecino_arriba_derecha = -1
            #     if(not np.isnan(um[j, i, 1])):    
            #         vecino_arriba_derecha = norm(um[j, i, 1])
            #     vecino_arriba_izquierda = -1
            #     if(not np.isnan(um[j, i, 2])):
            #         vecino_arriba_izquierda = norm(um[j, i, 2])              
            #     # Modificar los valores en la matriz_resultante
            #     matriz_resultante[2 * j, 2 * i] = valor_actual  # Valor actual
            #     matriz_resultante[2 * j, 2 * i + 1] = vecino_derecha  # Vecino a la derecha
            #     matriz_resultante[2 * j + 1, 2 * i + 1] = vecino_arriba_derecha  # Vecino arriba izquierda
            #     matriz_resultante[2 * j + 1, 2 * i - 1] = vecino_arriba_izquierda # Vecino arriba derecha


            # if(j==0 and i==0):
            #     valor_actual = norm(umat[j][i])
            #     vecino_derecha = -1
            #     if(not np.isnan(um[j, i, 0])):
            #         vecino_derecha = norm(um[j, i, 0])
            #     vecino_arriba_derecha = -1
            #     if(not np.isnan(um[j, i, 1])):    
            #         vecino_arriba_derecha = norm(um[j, i, 1])
            #     vecino_arriba_izquierda = -1
            #     if(not np.isnan(um[j, i, 2])):
            #         vecino_arriba_izquierda = norm(um[j, i, 2])              
            #     # Modificar los valores en la matriz_resultante
            #     matriz_resultante[2 * j, 2 * i] = valor_actual  # Valor actual
            #     matriz_resultante[2 * j, 2 * i + 1] = vecino_derecha  # Vecino a la derecha
            #     matriz_resultante[2 * j - 1, 2 * i] = vecino_arriba_derecha  # Vecino arriba izquierda
            #     matriz_resultante[2 * j + 1, 2 * i - 1] = vecino_arriba_izquierda # Vecino arriba derecha
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

def tuplas_hits(som_test):
    bmus = som_test._bmu[0].astype(int)
    unique, counts = np.unique(bmus, return_counts=True)
    unique = unique.astype(int).tolist()
    counts = counts.astype(int).tolist()
    return dict(zip(unique, counts))

def headers(self):
    self.send_header('Access-Control-Allow-Origin', '*')  # Permitir cualquier origen
    self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    self.send_header('Access-Control-Max-Age', '1000')
    self.send_header('Access-Control-Allow-Headers', '*')
    return self

def ok200(self):
    self.send_response(200)  # HTTP 200 OK response
    self.send_header('Content-type', 'application/json')  # Set the response content type to JSON
    self = headers(self)
    self.end_headers()
    return self
    
def error404(self):
    self.send_response(404)  
    self = headers(self)
    self.end_headers()
    return self

def error400(self, mensaje_error):
    self.send_response(400)
    self.send_header('Content-type', 'text/plain')
    self.end_headers()
    self.wfile.write(bytes("Error: %s" %mensaje_error, 'utf-8'))
    self.wfile.flush()

def json_return(datos, self):
    print("jsonnnnnnnn")
    with open('archivo.json', 'r') as file:
        datos = json.load(file)
    datos_json = json.dumps(datos)
    self = ok200(self)
    self.wfile.write(datos_json.encode())
    self.wfile.flush()

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

def bmu_return(datos,params,etiquetas,self):
    json_data = json.loads(datos)
    try:
        data = procesarJSON(json_data)
    except Exception as e: #Error al validar datos! Hay que avisar
       self = error400(self,str(e))
    
    # ENTRENO 
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

    #Cargo el ultimo entrenamiento en el json
    # with open("resultadoPrueba.json", "w") as file:
    #     json.dump(jsondata, file)
    jsondata = json.dumps(jsondata)
    self = ok200(self)
    self.wfile.write(jsondata.encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush() 

def cluster_return(datos,codebook,params,self):
    filas = int(params['filas'])
    columnas = int(params['columnas'])
    cant_clusters = int(params['cantidadClusters'])
    datos = procesarJSON(datos)
    resultado_clustering = kmeans(datos,codebook,filas,columnas,k=cant_clusters)
    self = ok200(self)
    self.wfile.write((json.dumps(resultado_clustering.tolist())).encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush() 

def nuevosdatos_return(datos,nuevosDatos,etiquetas, codebook,self):
    nuevosDatos = json.loads(nuevosDatos)
    nuevosDatos = [[float(value) for value in entry.values()] for entry in nuevosDatos]
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
    
    self = ok200(self)
    self.wfile.write(jsondata.encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush() 

def bmu_prueba(datos,params,self):
    # Abre el archivo JSON y lee su contenido
    with open('resultadoPrueba.json', 'r') as archivo:
        contenido_json = archivo.read()
    # cadena_json = json.dumps(contenido_json)
    self = ok200(self)
    self.wfile.write(contenido_json.encode())
    self.wfile.flush()
    

def switch_case(path, params,datos,etiquetas,codebook,nuevosDatos,self):
    
    if (path == '/bmu'):
        bmu_return(datos,params,etiquetas,self)
    elif (path == '/rapida'): 
        bmu_prueba(datos,params,self)
    elif (path == '/clusters'): 
        cluster_return(datos,codebook,params,self)
    elif (path == '/nuevosDatos'): 
        nuevosdatos_return(datos,nuevosDatos,etiquetas, codebook,self)
    else:
        error404(self)
 

class MyRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)  # agarramos el body de la request
        
        datos_de_entrada = json.loads(post_data)
        
        json_data = datos_de_entrada.get("datos", {})
        etiquetas = datos_de_entrada.get("etiquetas", {})
        codebook = datos_de_entrada.get("codebook", {})
        nuevosDatos = datos_de_entrada.get("nuevosDatos", {})
        #tipo = datos_de_entrada.get("tipo", "")
        params = datos_de_entrada.get("params", {})
        switch_case(self.path,params,json_data,etiquetas,codebook,nuevosDatos,self)

# Create an instance of the server with the request handler
server = socketserver.ThreadingTCPServer((host, port), MyRequestHandler)


# Start the server
print(f"Server running on {host}:{port}")
os.makedirs('Results', exist_ok=True)
server.serve_forever()
