from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index4.html")


@socketio.on("message_from_client")
def handle_message(message):
    print("Received message from client:", message)
    # 여기서 어떤 처리를 해도 됩니다.
    socketio.emit("message_from_server", "Server received your message: " + message)


if __name__ == "__main__":
    socketio.run(app, debug=True)
