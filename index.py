# import re
import os
import sys

# import sys
import google_stream_stt as gs  # stt 모듈
import numpy as np
from threading import Thread


# from keras import layers, models, optimizers, losses, metrics

# from tensorflow.keras.preprocessing.sequence import pad_sequences


# 사용자 모듈
import textCussDetect as td
import OpenCV_training as ot
import Varable as v
import flask_predict as fp

# 특정 폴더 경로 설정

# Path for face image database
data_path = v.data_path
training_path = v.training_path

RATE = 44100


# 조건을 만족할 때 삐- 소리 생성
def beep():
    frequency = 1000  # 소리의 높낮이
    duration = 1  # 지속시간 (초)

    t = np.arange(int(RATE * duration)) / RATE
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    return wave.astype(np.int16).tobytes()


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
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
