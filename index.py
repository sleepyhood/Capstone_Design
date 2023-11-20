import re
import os
import sys
import google_stream_stt as gs  # stt 모듈
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm

import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from konlpy.tag import Kkma
from keras import layers, models, optimizers, losses, metrics

# from tensorflow.keras.preprocessing.sequence import pad_sequences
import textCussDetect as td

from google.cloud import speech
import pyaudio
import keyboard
import cv2

# import opencv_predict as op
import temp

# 특정 폴더 경로 설정

# Path for face image database
data_path = "OpenCV_dataset"
training_path = "OpenCV_trainer"
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
# print(os.path.dirname(__file__))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# gs.main()
p = pyaudio.PyAudio()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


# 입력 스트림 열기
"""
stream_in = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)
"""
# 출력 스트림 열기
stream_out = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK
)


# 조건을 만족할 때 삐- 소리 생성
def beep():
    frequency = 1000  # 소리의 높낮이
    duration = 1  # 지속시간 (초)

    t = np.arange(int(RATE * duration)) / RATE
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    return wave.astype(np.int16).tobytes()


"""
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
"""

# td.training() #필요시 실행
# td.sentiment_predict("안녕하세요", tokenizer)

""""""


def googleSTT():
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

    # start bidirectional streaming from microphone input to speech API
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ko-KR",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = gs.ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:
        # while True:
        while not stream.closed and cam.isOpened():
            """
            # 추가중
            ret, img = cam.read()
            # img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save the captured image into the datasets folder
            if ret:
                cv2.imshow("image", img)
            k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
            if k == 27:
                break
            """
            # input_data = stream_in.read(CHUNK)
            # input_array = np.frombuffer(input_data, dtype=np.int16)

            # output_data = input_array.tobytes()
            # if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
            #     print("프로그램을 종료합니다.")
            #     break
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()

            # input_array = np.frombuffer(audio_generator, dtype=np.int16)  # 추가
            # output_data = audio_generator.tobytes()  # 추가

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)
            # print()
            # Now, put the transcription responses to use.
            sttResult = gs.listen_print_loop(responses, stream)
            print(f"sttResult: {sttResult}")

            #
            boolResult = td.sentiment_predict(sttResult, tokenizer)
            # if boolResult:
            #     output_data += beep()

            # 출력 스트림으로 데이터 전송
            # stream_out.write(output_data)  # 추가

            # listen_print_loop 함수로 결과가 출력됨
            # print(f"responses: {listen_print_loop(responses, stream)}")
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

            """
                출력 부분 추가
            """
            # 마이크에서 데이터 읽기
            print(f"stream.audio_input: {type(stream.audio_input)}")
        #    input_data = np.array(stream.audio_input)  # ?
    #     input_array = np.frombuffer(input_data, dtype=np.int16)

    # 여기에서 음성 처리 및 분석을 수행할 수 있습니다.
    # 이 예제에서는 받은 데이터를 그대로 출력으로 전달합니다.
    #      output_data = input_array.tobytes()
    #       stream_out.write(output_data)


"""
def main():
    try:
        while True:
            # 마이크에서 데이터 읽기
            input_data = stream_in.read(CHUNK_SIZE)
            input_array = np.frombuffer(input_data, dtype=np.int16)

            # 여기에서 음성 처리 및 분석을 수행할 수 있습니다.
            # 이 예제에서는 받은 데이터를 그대로 출력으로 전달합니다.
            output_data = input_array.tobytes()

            # googleSTT 함수 호출 시 입력 및 출력 스트림을 전달
            googleSTT()

            # 특정 조건을 만족하면 "삐-" 소리 추가
            # if True:
            #     output_data += beep()

            # 출력 스트림으로 데이터 전송
            stream_out.write(output_data)

    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
    finally:
        # 스트림 닫기
        stream_in.stop_stream()
        stream_in.close()
        stream_out.stop_stream()
        stream_out.close()
        p.terminate()
"""

# if __name__ == "__main__":
#     main()


# stt 호출
# googleSTT()
# cam.release()
# cv2.destroyAllWindows()


# 폴더
def list_files(folder_path):
    files = os.listdir(folder_path)
    print(f"\n{len(files)} Files in the folder:")
    for file in files:
        print(file)
        key = int(file.split(".")[1])  # 파일명에서 키 추출
        value = file.split(".")[2]  # 파일명에서 값 추출
        face_dict[key] = value


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


# Path for face image database
data_path = "OpenCV_dataset"
training_path = "OpenCV_trainer"

face_dict = {
    -1: "None",
}

# googleSTT()

list_files(training_path)
while True:
    print("\nOptions:")
    print("1. 이미지 촬영 및 학습")
    print("2. 학습 데이터 확인")
    print("3. 학습 데이터 제거")
    print("4. 웹캠 실행(+오디오 검열)")
    print("0. 종료")

    choice = input("Enter your choice (1/2/3/4): ")

    if choice == "0":
        print("Exiting the program.")
        break
    elif choice == "1":
        face_id, face_name = temp.training(data_path, training_path)
        face_dict[id] = face_name
    elif choice == "2":
        list_files(training_path)
    elif choice == "3":
        file_to_delete = input("Enter the file name to delete: ")
        file_path = os.path.join(training_path, file_to_delete)
        delete_file(file_path)
    elif choice == "4":
        temp.predict(face_dict, training_path)
    elif choice == "0":
        print("Exiting.")
    else:
        print("Invalid choice.")
