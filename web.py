from rwkv import Model, Tokenizer

from flask import Flask
from flask_sock import Sock
import umsgpack

app = Flask(__name__)
sock = Sock(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@sock.route('/ws')
def echo(ws):
    ws.send(umsgpack.packb("Hello"))
