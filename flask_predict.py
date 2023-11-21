from flask import Flask, render_template, Response
import cv2
import os
import threading
import Varable as v

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    print("INIT")

app = Flask(__name__)
training_path = v.training_path  # Replace with the actual path

# face_dict = v.face_dict
# face_dict = {
#     -1: "None",
# }
# print(f"flask_predict: {v.face_dict}")

# def list_files(folder_path):
#     files = os.listdir(folder_path)
#     print(f"\n{len(files)} Files in the folder:")
#     for file in files:
#         print(file)
#         key = int(file.split(".")[1])  # 파일명에서 키 추출
#         value = file.split(".")[2]  # 파일명에서 값 추출
#         face_dict[key] = value


# list_files(training_path)


def generate_frames():
    cam = cv2.VideoCapture(0)
    # cam.set(3, 640)  # set video width
    # cam.set(4, 480)  # set video height
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    for filename in os.listdir(training_path):
        if filename.endswith(".yml"):
            file_path = os.path.join(training_path, filename)
            recognizer.read(file_path)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y : y + h, x : x + w])
            print(f"id: {id}")
            # print(f"generate_frames: {v.face_dict}")  # 추가
            if confidence < 100 and id in v.face_dict:
                id = v.face_dict[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                roi = img[y : y + h, x : x + w]
                blurred_roi = cv2.GaussianBlur(roi, (75, 75), 0)
                img[y : y + h, x : x + w] = blurred_roi

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(
                img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1
            )

        _, jpeg = cv2.imencode(".jpg", img)
        frame = jpeg.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")  # /stream endpoint를 추가
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_app():
    app.run(debug=True, threaded=True)
    # app.run(host=v.my_ip, port="9080", debug=True)


if __name__ == "__main__":
    run_app()

# if __name__ == "__main__":
#     # Run Flask app in the main thread
#     my_ip = "172.16.20.122"
#     app.run(host=my_ip, port="9080", debug=True)
# my_ip = "172.16.20.122"
# app.run(host=v.my_ip, port="9080", debug=True)
