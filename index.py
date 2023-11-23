# import re
import os
import sys

# import sys
import google_stream_stt as gs  # stt 모듈
import numpy as np
from threading import Thread

# import pandas as pd
# import matplotlib.pyplot as plt

# import urllib.request
# from tqdm import tqdm

# import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from konlpy.tag import Kkma

# from keras import layers, models, optimizers, losses, metrics

# from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.cloud import speech
import pyaudio
import cv2

# 사용자 모듈
import textCussDetect as td
import OpenCV_training as ot
import Varable as v
import flask_predict as fp

# 특정 폴더 경로 설정

# Path for face image database
data_path = v.data_path
training_path = v.training_path

# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set video width
# cam.set(4, 480)  # set video height

# Audio recording parameters
# STREAMING_LIMIT = 240000  # 4 minutes
# SAMPLE_RATE = 16000
# CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

# RED = "\033[0;31m"
# GREEN = "\033[0;32m"
# YELLOW = "\033[0;33m"
# print(os.path.dirname(__file__))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# gs.main()
# p = pyaudio.PyAudio()

# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
RATE = 44100


# 입력 스트림 열기
"""
stream_in = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)
"""
# 출력 스트림 열기
# stream_out = p.open(
#     format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK
# )


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

# td.training()  # 필요시 실행
# td.sentiment_predict("안녕하세요", tokenizer)


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


# 폴더
def list_files(folder_path):
    files = os.listdir(folder_path)
    print(f"\n{len(files)} Files in the folder:")
    local_face_dict = {}
    for file in files:
        # print(file)
        key = int(file.split(".")[0])  # 파일명에서 키 추출
        value = file.split(".")[1]  # 파일명에서 값 추출
        local_face_dict[key] = value
    return local_face_dict


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


# 토큰나이저 추출
# def inintTextDetect():
#     print(f"index.py 에서 inintTextDetect() 호출")
#     (
#         X_train,
#         y_train,
#         X_test,
#         y_test,
#         tokenizer,
#         train_data,
#         test_data,
#         vocab_size,
#     ) = td.load_and_preprocess_data()
#     return tokenizer


v.face_dict = list_files(data_path)
print(f"index_v.face_dict: {v.face_dict}")


# !중요: token이 있어야 텍스트 정규화가 가능하므로 처음 한 번은 실행하고 predict할 때도 필요


def main():
    while True:
        print("\nOptions:")
        print("1. 이미지 촬영")
        print("2. 이미지 학습")
        print("3. 학습 데이터 확인")
        print("4. 웹캠 실행(+오디오 검열)")
        print("0. 종료")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "0":
            print("Exiting the program.")
            break
        elif choice == "1":
            face_id, face_name = ot.dataCollect(data_path)
            v.face_dict[face_id] = face_name
        elif choice == "2":
            ot.training(data_path, training_path)
        elif choice == "3":
            v.face_dict = list_files(data_path)
            print(v.face_dict)
        elif choice == "4":
            fp.run_app()
        else:
            print("Invalid choice.")


# textDetect_thread = Thread(target=inintTextDetect)


# textDetect_thread.daemon = True

# textDetect_thread.start()
# token = textDetect_thread.join()
# token = inintTextDetect()
main()
# main_thread = Thread(target=main, args=(token))
# main_thread.daemon = False

# main_thread.start()
# main_thread.join()
