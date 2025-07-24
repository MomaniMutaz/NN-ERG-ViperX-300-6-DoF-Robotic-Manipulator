# server.py (Windows)
import matlab.engine #library allows  to start and interact with a MATLAB engine session from Python.
import socket # this module allows to manage network connections

# this function starts a MATLAB session and returns an engine object that  can be used to interact with MATLAB.
eng = matlab.engine.start_matlab()

# Set up a TCP/IP server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #  This creates a new socket using the Internet address family (IPv4) and the TCP protocol.
#socket.AF_INET specifies the address family for IPv4.
#socket.SOCK_STREAM indicates that this is a TCP socket for stream-oriented connections.
server_socket.bind(('0.0.0.0', 12345))  # This binds the socket to the address '0.0.0.0' (This means that the server will accept incoming connections on any of the machine's IP addresses) and port 12345.
server_socket.listen(5) # This tells the socket to start listening for incoming connections. The 5 specifies the maximum number of queued connections.
print("Server listening on port 12345")

while True: #This creates an infinite loop to continuously accept and handle client connections.
    client_socket, addr = server_socket.accept() #This waits for an incoming connection. When a client connects, it returns a new socket object (client_socket) and the address of the client (addr).
    print(f"Connection from {addr}")
    
    data = client_socket.recv(4096).decode()  # This receives up to 4096 bytes of data from the client and decodes it from bytes to a string.
    if data: #This checks if any data was received
        try:
            # Parse the received data
            args = list(map(float, data.split(','))) #This splits the received data string by commas, converts each piece to a float, and stores the results in a list called args.
            
            # Call the MATLAB function with the parsed arguments
            result = eng.Mass_Inertia_Matrix_function(*args)
            
            # Format the result matrix
            result_str = '\n'.join(','.join(map(str, row)) for row in result)
            
            # Send the response back to the client
            client_socket.send(result_str.encode())
        except Exception as e: # If an error occurs during processing, it catches the exceptio
            client_socket.send(f"Error: {e}".encode())
    
    client_socket.close()



