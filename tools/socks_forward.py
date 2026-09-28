"""Forward a local TCP port to a tailnet host through tailscale's SOCKS5 proxy.

This host runs tailscale in userspace-networking mode, so tailnet peers are reachable only via
the SOCKS5 proxy on localhost:1080. Clients that speak plain HTTP with no proxy support (the
pipeline's urllib ChatClient) connect to the local port instead.

  python3 tools/socks_forward.py --listen 30000 --target spark:30000
"""
import argparse
import socket
import threading

import socks


def pipe(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client, host, port):
    upstream = socks.socksocket()
    upstream.set_proxy(socks.SOCKS5, "127.0.0.1", 1080, rdns=True)
    try:
        upstream.connect((host, port))
    except OSError:
        client.close()
        return
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", required=True, help="host:port on the tailnet")
    args = ap.parse_args()
    host, port = args.target.rsplit(":", 1)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.listen))
    srv.listen(64)
    while True:
        client, _ = srv.accept()
        threading.Thread(target=handle, args=(client, host, int(port)), daemon=True).start()


if __name__ == "__main__":
    main()
