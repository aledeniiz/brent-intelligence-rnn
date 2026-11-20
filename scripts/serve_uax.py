"""Serve the local HTML product, bound exclusively to127.0.0.1."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,json
from brent_intelligence.uax_product import Product,ROOT

def handler(product):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,obj,status=200):
            content=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(content)
        def allowed_host(self):return self.headers.get('Host') in ('127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port))
        def do_GET(self):
            if not self.allowed_host():return self.send_json({'error':'Host no permitido'},403)
            try:
                if self.path=='/api/health':return self.send_json({'status':'ok','training_endpoint':False})
                if self.path=='/api/state':return self.send_json(product.state())
                if self.path in ('/','/index.html'):file=ROOT/'web/uax/index.html';mime='text/html; charset=utf-8'
                elif self.path=='/assets/plotly.min.js':
                    import plotly
                    file=Path(plotly.__file__).parent/'package_data/plotly.min.js';mime='application/javascript'
                else:return self.send_json({'error':'Ruta no encontrada'},404)
                content=file.read_bytes();self.send_response(200);self.send_header('Content-Type',mime)
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(content)
            except Exception as exc:self.send_json({'status':'blocked','horizons':[],'error':str(exc)},422)
        def do_POST(self):
            expected='http://'+self.headers.get('Host','')
            if (not self.allowed_host() or self.path!='/api/refresh' or self.headers.get('Origin')!=expected
                or self.headers.get('X-UAX-Action')!='refresh' or self.headers.get('Content-Type')!='application/json'):
                return self.send_json({'error':'Acción local no autorizada'},403)
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.send_json({'error':'Longitud inválida'},400)
            if length<0:return self.send_json({'error':'Longitud inválida'},400)
            if length>1024:return self.send_json({'error':'Solicitud demasiado grande'},413)
            try:
                if json.loads(self.rfile.read(length) or b'{}')!={}:raise ValueError('No se admiten parámetros de rutas o entrenamiento')
                result=product.refresh();self.send_json(result,200 if result['status']=='ready' else 422)
            except Exception as exc:self.send_json({'status':'blocked','horizons':[],'error':str(exc)},422)
        def log_message(self,*args):pass
    return Handler

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(Product()))
    print(f'Brent Intelligence HTML: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
