# app.py

from flask import Flask, render_template, Response
from time import sleep
import cv2

app = Flask(__name__)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate_frames():
    while True:
        sleep(0.1)
        ref, frame = capture.read()
        if not ref:
            break
        else:
            ref, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_app():
    my_ip = "172.16.20.122"
    app.run(host=my_ip, port="9080")


# if __name__ == "__main__":
#     run_app()


# 아래가 브라우저 주소
# http://172.16.20.122:9080/
# http://172.16.20.122:9080/stream
