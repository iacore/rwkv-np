from flask import Flask
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@sock.route('/ws')
def echo(ws):
    ws.send("Hello")
