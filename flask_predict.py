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
import winsound as sd


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
# 아스키값 false 해야 한글 출력
training_path = v.training_path  # Replace with the actual path

# global 변수 앞에 Lock 추가
transcript_result = ""
cussCount = 0  # 추가: 욕설 카운트 변수

transcript = ""

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
WHITE = "\033[0;37m"


def beepsound():
    fr = 2000  # range : 37 ~ 32767
    du = 1000  # 1000 ms ==1second
    sd.Beep(fr, du)  # winsound.Beep(frequency, duration)


# ====================비디오====================
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


# ====================비디오====================


# ====================오디오====================
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
            transcript_result = gs.listen_print_loop(responses, stream)

            print(transcript_result)
            # print(td.cuss_predict(transcript_result))
            tempSen = transcript_result
            tempCnt = cussCount
            # not 으로 바꾸면 욕설의 여부 결과가 반대가 됨
            if td.cuss_predict(transcript_result):
                sys.stdout.write(YELLOW)
                sys.stdout.write("욕설 감지")
                sys.stdout.write(WHITE)
                cussCount += 1  # 추가: 욕설이 감지되면 cussCount 증가
                tempCnt = cussCount
                # 삐 소리
                beepsound()

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
                return tempSen, tempCnt
            # 조건없이 return시 실행 안됨
            # return transcript_result, cussCount


# ====================오디오====================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcript", methods=["GET"])
def get_transcript():
    global transcript_result, cussCount
    sen = ""
    cnt = 0
    # start_google_stt 함수 내부 while 존재
    sen, cnt = start_google_stt()

    # transcript_result, cuss_count = start_google_stt()

    # if not transcript_queue.empty():
    #     transcript_result, cuss_count = transcript_queue.get()
    #     cussCount = cuss_count
    # transcript_result, cussCount = start_google_stt()

    return jsonify({"transcript_result": sen, "cussCount": cnt})


@app.route("/stream")  # /stream endpoint를 추가
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_app():
    print("run_app호출")

    # !중요: token이 있어야 텍스트 정규화가 가능하므로 처음 한 번은 실행하고 predict할 때도 필요
    textDetect_thread = Thread(target=td.load_and_preprocess_data)

    textDetect_thread.start()
    textDetect_thread.join()

    app.run(debug=True, threaded=True, use_reloader=False)


#  * Running on http://127.0.0.1:5000
# 이 코드가 있으면 import 하면 자동 실행되므로 주의
# if __name__ == "__main__":
#     run_app()
