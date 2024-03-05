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

# Define the server address and port
host = "localhost"
port = 7777


def procesarJSON(data): #Validar que el dataframe sea válido! O que lo haga dart, una de las dos
    df = pd.DataFrame(data)
    df.set_index(df.columns[0], inplace=True) #Importante, esto le marca que la primera columna no son datos, sino que es la etiqueta/nombre
    df = df.astype(float)
    print(df)
    return df

def train(data,params):
    fil = int(params["filas"])
    col = int(params["columnas"])  #ojo que en el front aun no está validado que sean numeros, verificar.
    fvecindad = params["vecindad"]
    inicializa = params["inicializacion"]
    itera = int(params["iteraciones"])
    normalizacion = params["normalizacion"]
    trainLenFactor = int(params["trainLengthFactor"])
    if normalizacion == "None":
        normalizacion = None
    mapsize = (col,fil)
    som_test = intrasom.SOMFactory.build(data,
        #mask=-9999,
        mapsize=mapsize,
        mapshape='planar',
        lattice='hexa',
        normalization=normalizacion,
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
    som_test.train(maxtrainlen=itera,train_len_factor=trainLenFactor, previous_epoch = True)

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
            valor_actual = norm(umat[j][i])
            vecino_derecha = -1
            if(not np.isnan(um[j, i, 0])):
                vecino_derecha = norm(um[j, i, 0])
            vecino_abajo_derecha = -1
            if(not np.isnan(um[j, i, 1])):    
                vecino_abajo_derecha = norm(um[j, i, 1])
            vecino_abajo_izquierda = -1
            if(not np.isnan(um[j, i, 2])):
                vecino_abajo_izquierda = norm(um[j, i, 2])              
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

def ok200(self):
    self.send_response(200)  # HTTP 200 OK response
    self.send_header('Content-type', 'application/json')  # Set the response content type to JSON
    self.send_header('Access-Control-Allow-Origin', '*')  # Permitir cualquier origen
    self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    self.send_header('Access-Control-Max-Age', '1000')
    self.send_header('Access-Control-Allow-Headers', '*')
    self.end_headers()
    return self
    

def json_return(datos,self):
    print("jsonnnnnnnn")
    with open('archivo.json', 'r') as file:
        datos = json.load(file)
    datos_json = json.dumps(datos)
    self = ok200(self)
    self.wfile.write(datos_json.encode()) 
    self.wfile.flush()

def bmu_return(datos,params,self):
    print("bmuuuuu")
    json_data = json.loads(datos)
    data = procesarJSON(json_data)  
    resultados_entrenamiento = train(data,params) #TODO los parametros de entrenamiento hay que pasarlos en realidad, esta todo default
    
    # prueba hits
    resultado_hits = tuplas_hits(resultados_entrenamiento)
    json.dumps(resultado_hits)


    resultado_umat = tuplas_umat(resultados_entrenamiento)
    json.dumps(resultado_umat, indent = 2)
   
    resultados_entrenamiento = resultados_entrenamiento.neurons_dataframe
    resultados_entrenamiento = pd.DataFrame.to_json(resultados_entrenamiento)
    resultados_entrenamiento = json.dumps(resultados_entrenamiento)
    
    jsondata = {}
    jsondata['Neurons'] = resultados_entrenamiento
    jsondata['UMat'] = resultado_umat

    # prueba hits
    jsondata['Hits'] = resultado_hits
 
    jsondata = json.dumps(jsondata)
    jsondata = jsondata.replace('\\','') #ESTO NO LO PUDE ARREGLAR DE OTRA FORMA. (Funciona OK de todas formas)
    jsondata = jsondata.replace('""','') #El JSON tiene caracteres extraños/malformados, los elimine asi, pero probablemente sea un arraste de error de algo anterior.
    
    self = ok200(self)
    self.wfile.write(jsondata.encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush() 

def bmu_returnSinEntrenar(datos,params,self):
    print("bmuuuuu")
    # json_data = json.loads(datos)
    # data = procesarJSON(json_data)  
    # som_test = train_som_test(data)
    # resultados_bmu = som_test.neurons_dataframe
    # resultados_bmu.to_csv('resultados_bmu.csv', index=False) ###
    # resultado_umat = tuplas_umat(som_test)
    # with open('resultado_umat.json', 'w') as archivo_json: ###
    #     json.dump(resultado_umat, archivo_json, indent=2) ###
    resultados_bmu = pd.read_csv('resultados_bmu.csv') 
    print(resultados_bmu)
    with open('resultado_umat.json', 'r') as archivo_json:
        resultado_umat = json.load(archivo_json)
    #TODO Dejar de leer de archivo y que entrene
    resultados_bmu_json = pd.DataFrame.to_json(resultados_bmu)
    resultados_bmu_json = json.dumps(resultados_bmu_json)
    
    jsondata = {}
    jsondata['Neurons'] = resultados_bmu_json
    jsondata['UMat'] = resultado_umat
 
    jsondata = json.dumps(jsondata)
    jsondata = jsondata.replace('\\','') #ESTO NO LO PUDE ARREGLAR DE OTRA FORMA. (Funciona OK de todas formas)
    jsondata = jsondata.replace('""','') #El JSON tiene caracteres extraños/malformados, los elimine asi, pero probablemente sea un arraste de error de algo anterior.
    
    # print(json.dumps(jsondata))
    self = ok200(self)
    self.wfile.write(jsondata.encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush()

# def umat_return(datos,self):
#     json_data = json.loads(datos)
#     data = procesarJSON(json_data)
#     som_test = train(data)
#     som_test = som_test.neurons_dataframe
#     result = tuplas_umat(som_test)
#     resultados_entrenamiento = pd.DataFrame.to_json(result)
#     self = ok200(self)
#     self.wfile.write(resultados_entrenamiento.encode())  # Send the resultados_entrenamiento JSON as the response
#     self.wfile.flush()

def default():
    print("Ejecutando caso por defecto")

def switch_case(path, params,datos,self):
    switch_dict = {
        '/json': bmu_returnSinEntrenar,
        '/bmu': bmu_return,
        # '/umat': umat_return
    }
    switch_dict.get(path, default)(datos,params,self)

class MyRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)  # agarramos el body de la request
        
        datos_de_entrada = json.loads(post_data)
        
        json_data = datos_de_entrada.get("datos", {})
        tipo = datos_de_entrada.get("tipo", "")
        params = datos_de_entrada.get("params", {})
        switch_case(self.path,params,json_data, self)

# Create an instance of the server with the request handler
server = socketserver.TCPServer((host, port), MyRequestHandler)


# Start the server
print(f"Server running on {host}:{port}")
os.makedirs('Results', exist_ok=True)
server.serve_forever()

#****************************************************************
#****************************************************************
#****************************************************************


# dfejemplos = pd.read_json("datos.json")
# print(df)





        # if tipo == "json":
        #     print("jsonnnnnnnn")
        #     with open('archivo.json', 'r') as file:
        #         datos = json.load(file)
        #     datos_json = json.dumps(datos)
        #     self.send_response(200)  # HTTP 200 OK response
        #     self.send_header('Content-type', 'application/json')  # Set the response content type to JSON
        #     self.send_header('Access-Control-Allow-Origin', '*')  # Permitir cualquier origen
        #     self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        #     self.send_header('Access-Control-Max-Age', '1000')
        #     self.send_header('Access-Control-Allow-Headers', '*')
        #     self.end_headers()
        #     self.wfile.write(datos_json.encode()) 
        #     self.wfile.flush()
        # else:          

        #     data = procesarJSON(json_data)  
        #     resultados_entrenamiento = train(data) #TODO los parametros de entrenamiento hay que pasarlos en realidad, esta todo default
        #     resultados_entrenamiento = pd.DataFrame.to_json(resultados_entrenamiento)

        #     self.send_response(200)  # HTTP 200 OK response
        #     self.send_header('Content-type', 'application/json')  # Set the response content type to JSON
        #     self.send_header('Access-Control-Allow-Origin', '*')  # Permitir cualquier origen
        #     self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        #     self.send_header('Access-Control-Max-Age', '1000')
        #     self.send_header('Access-Control-Allow-Headers', '*')
        #     self.end_headers()
        #     # self.wfile.write(resultados_entrenamiento.encode())  # Send the resultados_entrenamiento JSON as the response
        #     self.wfile.write(datos_json.encode()) 
        #     self.wfile.flush()