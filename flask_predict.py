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

# if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
#     print("INIT")

app = Flask(__name__)
training_path = v.training_path  # Replace with the actual path
transcript_result = ""
transcript_queue = queue.Queue()  # 추가: STT 결과를 저장할 큐

X_train, y_train, X_test, y_test, tokenizer, train_data, test_data, vocab_size = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)


def inintTextDetect():
    global X_train, y_train, X_test, y_test, tokenizer, train_data, test_data, vocab_size
    # 최초 1회 생성
    (
        X_train,
        y_train,
        X_test,
        y_test,
        tokenizer,
        train_data,
        test_data,
        vocab_size,
    ) = td.load_and_preprocess_data()


# 텍스트 기반 욕설 훈련을 재호출할 경우
def training():
    td.training()


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
    global transcript_result

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


# =====================비디오====================


# ======================오디오====================
def start_google_stt():
    global tokenizer
    while True:
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

        with mic_manager as stream:
            while not stream.closed:
                stream.audio_input = []
                audio_generator = stream.generator()

                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )

                responses = client.streaming_recognize(streaming_config, requests)
                global transcript_result
                transcript_result = gs.listen_print_loop(responses, stream)
                global transcript_queue
                transcript_queue.put(transcript_result)
                # 수정: STT 결과를 큐에 저장

                print(transcript_result)
                print(td.cuss_predict(transcript_result, tokenizer))
                if td.cuss_predict(transcript_result, tokenizer):
                    print("욕 멈춰")
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


@app.route("/")
def index():
    # return render_template("index.html", transcript=transcript_result)
    return render_template("index.html")


# @app.route("/transcript")
# def stream():
#     return Response(transcript_result, content_type="text/event-stream")


@app.route("/transcript")
def get_transcript():
    # 수정: 큐에서 STT 결과를 가져와 반환
    global transcript_queue
    global transcript_result
    if not transcript_queue.empty():
        transcript_result = transcript_queue.get()
    return Response(transcript_result, content_type="text/plain")
    # return jsonify({"transcript_result": transcript_result})

    # 아래는 1초 간격으로 업데이트 되므로 이전 레코드를 남겨야한다면 주석처리
    # return Response("No transcript available", content_type="text/plain")


@app.route("/stream")  # /stream endpoint를 추가
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_app():
    # stt_thread = Thread(target=start_google_stt)
    # stt_thread = Thread()
    # stt_thread.start()

    textDetect_thread = Thread(target=inintTextDetect)
    # print(f"token: {token}")
    stt_thread = Thread(target=start_google_stt)
    video_thread = Thread(target=generate_frames)

    # 메인 쓰레드 종료시 같이 종료 설정
    textDetect_thread.daemon = True
    stt_thread.daemon = True
    video_thread.daemon = True

    textDetect_thread.start()
    textDetect_thread.join()

    stt_thread.start()
    video_thread.start()

    # stt_thread.join()
    # video_thread.join()

    app.run(debug=True, threaded=True)
    # app.run(host=v.my_ip, port="9080", debug=True)


# 이 코드가 있으면 import 하면 자동 실행되므로 주의
# if __name__ == "__main__":
#     run_app()

# stt_thread = Thread(target=start_google_stt)
# stt_thread.start()
