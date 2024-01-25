import os
import pandas as pd
import intrasom
import http.server
import socketserver
import json
from intrasom.visualization import PlotFactory
import numpy as np
import matplotlib as mpl

# Define the server address and port
host = "localhost"
port = 7777


def procesarJSON(data): #Validar que el dataframe sea válido! O que lo haga dart, una de las dos
    df = pd.DataFrame(data)
    df.set_index(df.columns[0], inplace=True) #Importante, esto le marca que la primera columna no son datos, sino que es la etiqueta/nombre
    df = df.astype(float)
    print(df)
    return df
    
def train(data):
    mapsize = (24,14)
    som_test = intrasom.SOMFactory.build(data,
        mask=-9999,
        mapsize=mapsize,
        mapshape='toroid',
        lattice='hexa',
        normalization='var',
        initialization='random',
        neighborhood='gaussian',
        training='batch',
        name='Ejemplo',
        component_names=None,
        unit_names = None,
        sample_names=None,
        missing=True,
        save_nan_hist = True,
        pred_size=0)
    som_test.train(train_len_factor=2, previous_epoch = True)
    # print(som_test.results_dataframe)
    # return som_test.results_dataframe
    print(som_test.neurons_dataframe)
    return som_test.neurons_dataframe

def train_som_test(data):
    mapsize = (24,14)
    som_test = intrasom.SOMFactory.build(data,
        mask=-9999,
        mapsize=mapsize,
        mapshape='toroid',
        lattice='hexa',
        normalization='var',
        initialization='random',
        neighborhood='gaussian',
        training='batch',
        name='Ejemplo',
        component_names=None,
        unit_names = None,
        sample_names=None,
        missing=True,
        save_nan_hist = True,
        pred_size=0)
    som_test.train(train_len_factor=2, previous_epoch = True)

    return som_test

def tuplas_umat(som_test):
    plot = PlotFactory(som_test)
    # ruta completa para los archivos
    ruta_um = r'C:\Users\dell\Desktop\umatrix_expanded.txt'
    ruta_umat = r'C:\Users\dell\Desktop\umatrix_not_expanded.txt'
    um = plot.build_umatrix(expanded = True)
    umat = plot.build_umatrix(expanded = False)
    # Aplana las matrices tridimensionales
    um_flat = um.flatten()
    umat_flat = umat.flatten()
    # Guardar en archivos de texto con las rutas completas
    np.savetxt(ruta_um, um_flat, fmt='%f', delimiter='\t')
    np.savetxt(ruta_umat, umat_flat, fmt='%f', delimiter='\t')
    # Funcion para normalizar los valores
    norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
    # Matriz en la uqe voy a guardar los colores de las neuronas
    matriz_resultante = np.zeros((2 * som_test.mapsize[1], 2 * som_test.mapsize[0]))
    # Cada neurona me da cuatro valores, por lo que por cada una modico cuatro lugares en l amatriz
    for j in range(som_test.mapsize[1]):
        for i in range(som_test.mapsize[0]):
            valor_actual = norm(umat[j][i])
            vecino_derecha = norm(um[j, i, 0])
            vecino_arriba_derecha = norm(um[j, i, 1])
            vecino_arriba_izquierda = norm(um[j, i, 2])            
            # Modificar los valores en la matriz_resultante
            matriz_resultante[2 * j, 2 * i] = valor_actual  # Valor actual
            matriz_resultante[2 * j, 2 * i + 1] = vecino_derecha  # Vecino a la derecha
            matriz_resultante[2 * j + 1, 2 * i] = vecino_arriba_izquierda  # Vecino arriba izquierda
            matriz_resultante[2 * j + 1, 2 * i + 1] = vecino_arriba_derecha  # Vecino arriba derecha
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

def bmu_return(datos,self):
    print("bmuuuuu")
    json_data = json.loads(datos)
    data = procesarJSON(json_data)  
    resultados_entrenamiento = train(data) #TODO los parametros de entrenamiento hay que pasarlos en realidad, esta todo default
    resultados_entrenamiento = pd.DataFrame.to_json(resultados_entrenamiento)
    self = ok200(self)
    self.wfile.write(resultados_entrenamiento.encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush()

def umat_return(datos,self):
    json_data = json.loads(datos)
    data = procesarJSON(json_data)
    som_test = train_som_test(data)
    result = tuplas_umat(som_test)
    resultados_entrenamiento = pd.DataFrame.to_json(result)
    self = ok200(self)
    self.wfile.write(resultados_entrenamiento.encode())  # Send the resultados_entrenamiento JSON as the response
    self.wfile.flush()

def default():
    print("Ejecutando caso por defecto")

def switch_case(path, datos,self):
    switch_dict = {
        '/json': json_return,
        '/bmu': bmu_return,
        '/umat': umat_return
    }
    switch_dict.get(path, default)(datos,self)

class MyRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)  # agarramos el body de la request

        path = self.path # tipo de llamada

        datos_de_entrada = json.loads(post_data)
        
        json_data = datos_de_entrada.get("datos", {})
        tipo = datos_de_entrada.get("tipo", "")

        switch_case(path,json_data, self)
        
    #POR AHORA NO HAY NADA QUE USE GET, LO COMENTO
    # def do_GET(self): 
    #     # Set the response status code
    #     self.send_response(200)

    #     # Set the response headers
    #     self.send_header("Content-type", "text/html")
    #     self.send_header('Access-Control-Allow-Origin', '*')
    #     self.end_headers()

    #     # Send the response body
    #     self.wfile.write(b"<h1>Hello, World!</h1>")

# Create an instance of the server with the request handler
server = socketserver.TCPServer((host, port), MyRequestHandler)


# Start the server
print(f"Server running on {host}:{port}")
os.makedirs('Results', exist_ok=True)
server.serve_forever()



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