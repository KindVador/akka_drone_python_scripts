# -*- coding: utf-8 -*-
import socket
import struct


UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(8)    # buffer size is 8 bytes
    print("received message:", struct.unpack('ff', data))
