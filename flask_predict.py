from flask import Flask, render_template, Response, jsonify
import cv2
import os
import speech_recognition as sr
from google.cloud import speech
import google_stream_stt as gs
import pyaudio
import sys
import queue  # 추가
import textCussDetect as td  # 텍스트 욕설 검출

# import threading
import Varable as v
from threading import Thread
import threading

# if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
#     print("INIT")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
# 아스키값 false 해야 한글 출력
training_path = v.training_path  # Replace with the actual path

# global 변수 앞에 Lock 추가
transcript_result = ""
transcript_result_lock = threading.Lock()

transcript_queue = queue.Queue()  # 추가: STT 결과를 저장할 큐
cussCount = 0  # 추가: 욕설 카운트 변수

transcript = ""

# X_train, y_train, X_test, y_test, tokenizer, train_data, test_data, vocab_size = (
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
# )
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
WHITE = "\033[0;37m"


def start_streaming():
    while True:
        global transcript_result, cuss_count
        transcript_result, cuss_count = start_google_stt()
        transcript_queue.put((transcript_result, cuss_count))


# 텍스트 기반 욕설 훈련을 재호출할 경우
def training():
    td.training()


def generate_frames():
    cam = cv2.VideoCapture(0)
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
        # out(cam, face_detector, minW, minH, recognizer, font)
        # =====================비디오====================
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
            # print(f"id: {id}")
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


# =====================비디오====================


# ======================오디오====================
def start_google_stt():
    # transcript_result = ""
    global cussCount, transcript_result
    # global transcript_queue, cussCount, transcript_result
    # while True:
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = gs.ResumableMicrophoneStream(16000, 1024)
    sys.stdout.write("\nListening, say 'Quit' or 'Exit' to stop.\n\n")
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    tempSen = ""
    with mic_manager as stream:
        while not stream.closed:
            # while True:
            sys.stdout.write(GREEN)
            sys.stdout.write("\nwhile not stream.closed:\n")
            sys.stdout.write(WHITE)

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)
            # global transcript_result
            transcript_result = gs.listen_print_loop(responses, stream)
            # global transcript_queue
            transcript_queue.put(transcript_result)
            # 수정: STT 결과를 큐에 저장

            print(transcript_result)
            # print(td.cuss_predict(transcript_result))
            tempSen = transcript_result
            if td.cuss_predict(transcript_result):
                sys.stdout.write(YELLOW)
                sys.stdout.write("욕설 감지")
                sys.stdout.write(WHITE)
                cussCount += 1  # 추가: 욕설이 감지되면 cussCount 증가
                # 클라이언트에게 결과를 JSON 형태로 전송
                # return jsonify({"is_cuss": is_cuss})

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True

            # 문장을 하나라도 변환했다면 계속 스트림을 받지않고 출력해야함
            # 즉, 다시 웹으로 리턴해야함
            if tempSen != "":
                return tempSen, cussCount
            # return transcript_result, cussCount


@app.route("/")
def index():
    global cussCount, transcript_result

    # start_google_stt()
    # global transcript_result, cussCount
    # return render_template("index.html", transcript=transcript_result)
    # return render_template("index.html")
    return render_template("index.html")


@app.route("/transcript", methods=["GET", "POST"])
def get_transcript():
    sys.stdout.write(WHITE)
    # sys.stdout.write("get_transcript 호출됨")
    # global transcript_queue, cussCount, transcript_result
    global transcript_result, cussCount
    sen = ""
    # cnt = 0

    sen, cussCount = start_google_stt()
    # with transcript_result_lock:
    #     transcript_result, cuss_count = start_google_stt()
    # sen, cnt = start_google_stt()
    # transcript_result, cuss_count = start_google_stt()
    # cuss_count = int(cuss_count)
    # if not transcript_queue.empty():
    #     transcript_result = transcript_queue.get()
    # if not transcript_queue.empty():
    #     transcript_result, cuss_count = transcript_queue.get()
    #     cussCount = cuss_count
    # transcript_result, cussCount = start_google_stt()
    # print(f"in flask_predict: {sen}")
    return jsonify({"transcript_result": sen, "cussCount": cussCount})


@app.route("/stream")  # /stream endpoint를 추가
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def a():
    for i in range(50):
        print(i)


def run_app():
    print("run_app호출")

    textDetect_thread = Thread(target=td.load_and_preprocess_data)

    textDetect_thread.start()
    textDetect_thread.join()

    app.run(debug=True, threaded=True, use_reloader=False)


#  * Running on http://127.0.0.1:5000
# 이 코드가 있으면 import 하면 자동 실행되므로 주의
# if __name__ == "__main__":
#     run_app()
