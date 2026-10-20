"""
Simple local development web server for previewing the Prepix documentation site.
Usage:
    python website/serve.py [port]
"""
import http.server
import os
import socketserver
import sys

DEFAULT_PORT = 8080


def serve():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    web_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(web_dir)
    handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Prepix Documentation Server running at: http://localhost:{port}/")
            print("Press Ctrl+C to stop the server.")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error starting server on port {port}: {e}")
        print(f"Try running with an alternate port: python website/serve.py 8081")


if __name__ == "__main__":
    serve()
