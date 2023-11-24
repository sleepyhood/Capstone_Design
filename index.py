import os
import google_stream_stt as gs  # stt 모듈
import numpy as np
from threading import Thread

# 사용자 모듈
import textCussDetect as td
import OpenCV_training as ot
import Varable as v
import flask_predict as fp

# 특정 폴더 경로 설정

# Path for face image database
data_path = v.data_path
training_path = v.training_path


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


# 파일 삭제 함수
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


v.face_dict = list_files(data_path)
print(f"index_v.face_dict: {v.face_dict}")


def main():
    while True:
        print("\nOptions:")
        print("1. [비디오] 이미지 촬영")
        print("2. [비디오] 이미지 학습")
        print("3. [비디오] 학습된 인원 확인")
        print("4. [오디오] 텍스트 검열 모델 훈련(필요시 실행)")
        print("5. [비디오+오디오] 모자이크와 비속어 판별(웹에서 실행됩니다.)")
        print("0. 종료")

        choice = input("Enter your choice (1/2/3/4/5/0): ")

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
        elif choice == "4":  # 텍스트 기반 욕설 훈련을 재호출할 경우
            td.training()
            print("모델 재훈련 완료")
        elif choice == "5":
            print("토크나이저 초기화 후 실행됩니다...")
            fp.run_app()
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
