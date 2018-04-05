#!/usr/bin/env python
# Script to send paths to run in blender:
#   blender_client.py script1.py script2.py

PORT = 8081
HOST = "localhost"

def main():
    import sys
    import socket

    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect((HOST, PORT))

    for arg in sys.argv[1:]:
        clientsocket.sendall(arg.encode("utf-8") + b'\x00')


if __name__ == "__main__":
    main()