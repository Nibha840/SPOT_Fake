import http.server
import json
import base64
import io
import os
import sys
from PIL import Image
from predict import predict

PORT = 8080

class PredictionHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP request logging to keep the console clean
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.wfile.write(f"Error loading index.html: {e}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                image_b64 = data["image"]
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                image_data = base64.b64decode(image_b64)
                
                img = Image.open(io.BytesIO(image_data))
                
                # predict() is ready to process a PIL image object directly
                score = predict(img)
                
                response = {"success": True, "score": round(score, 4)}
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
            except Exception as e:
                response = {"success": False, "error": str(e)}
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=http.server.HTTPServer, handler_class=PredictionHandler):
    server_address = ("", PORT)
    httpd = server_class(server_address, handler_class)
    print(f"\n=======================================================")
    print(f" Live Demo Server running on: http://localhost:{PORT}")
    print(f" Press Ctrl+C to stop.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        sys.exit(0)

if __name__ == "__main__":
    run()
