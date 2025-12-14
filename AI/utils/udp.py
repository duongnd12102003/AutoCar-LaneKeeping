import socket

def make_rx(listen_ip: str, listen_port: int, rbuf_bytes: int = 8 * 1024 * 1024) -> socket.socket:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((listen_ip, listen_port))
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rbuf_bytes)
    return rx

def make_tx() -> socket.socket:
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return tx
