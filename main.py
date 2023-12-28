import pandas as pd
import intrasom
import http.server
import socketserver
import json

# Define the server address and port
host = "localhost"
port = 7777


def procesarJSON(data):
    df = pd.DataFrame(data)
    # Perform further processing on the DataFrame if needed
    print(df)
    return df
       
# Create a request handler class by subclassing http.server.BaseHTTPRequestHandler
class MyRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)  # Get the body of the request

        # Convert the POST data to JSON
        json_data = json.loads(post_data)

        print(json_data)
        data = procesarJSON(json_data)  # Call the defined function

        # mapsize = (24,14)
        # som_test = intrasom.SOMFactory.build(data,
        #                              mask=-9999,
        #                              mapsize=mapsize,
        #                              mapshape='toroid',
        #                              lattice='hexa',
        #                              normalization='var',
        #                              initialization='random',
        #                              neighborhood='gaussian',
        #                              training='batch',
        #                              name='Example',
        #                              component_names=None,
        #                              unit_names = None,
        #                              sample_names=None,
        #                              missing=True,
        #                              save_nan_hist = True,
        #                              pred_size=0)

        # som_test.train(train_len_factor=2,
        #        previous_epoch = True)

        



        self.send_response(200)  # HTTP 200 OK response
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')  # Permitir cualquier origen
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Max-Age', '1000')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(b'Recibido!')  # Send a response to the client
        self.wfile.flush()
    
    def do_GET(self):
        # Set the response status code
        self.send_response(200)

        # Set the response headers
        self.send_header("Content-type", "text/html")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Send the response body
        self.wfile.write(b"<h1>Hello, World!</h1>")

# Create an instance of the server with the request handler
server = socketserver.TCPServer((host, port), MyRequestHandler)


# Start the server
print(f"Server running on {host}:{port}")
server.serve_forever()



# dfejemplos = pd.read_json("datos.json")
# print(df)





