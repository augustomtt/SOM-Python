import os
import pandas as pd
import intrasom
import http.server
import socketserver
import json

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
    print(som_test.results_dataframe)
    return som_test.results_dataframe
class MyRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)  # agarramos el body de la request

        # pasar body a json
        json_data = json.loads(post_data)

        data = procesarJSON(json_data)  
        resultados_entrenamiento = train(data) #TODO los parametros de entrenamiento hay que pasarlos en realidad, esta todo default
        resultados_entrenamiento = pd.DataFrame.to_json(resultados_entrenamiento)

        self.send_response(200)  # HTTP 200 OK response
        self.send_header('Content-type', 'application/json')  # Set the response content type to JSON
        self.send_header('Access-Control-Allow-Origin', '*')  # Permitir cualquier origen
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Max-Age', '1000')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(resultados_entrenamiento.encode())  # Send the resultados_entrenamiento JSON as the response
        self.wfile.flush()
    
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





