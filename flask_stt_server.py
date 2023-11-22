from flask import Flask, request, jsonify
import json

app = Flask(__name__)


@app.route("/")
def home():
    return "hello world"


@app.route("/stt", methods=["POST"])
def stream():
    if request.method == "POST":
        content_type = request.headers.get("Content-Type")
        if content_type == "application/json":
            request_json = request.json
            return "ok"
        else:
            return "sth error"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
