import pyaudio
import numpy as np
import winsound as sd
import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
# import beepsound as bs

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

# 입력 스트림 열기
stream_in = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

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


print("마이크로 말하세요...")

try:
    while True:
        # 마이크에서 데이터 읽기
        input_data = stream_in.read(CHUNK)
        input_array = np.frombuffer(input_data, dtype=np.int16)

        # 여기에서 음성 처리 및 분석을 수행할 수 있습니다.
        # 이 예제에서는 받은 데이터를 그대로 출력으로 전달합니다.
        output_data = input_array.tobytes()

        # 특정 조건을 만족하면 "삐-" 소리 추가
        if False:
            output_data += beep()

        # 출력 스트림으로 데이터 전송
        stream_out.write(output_data)
        # print(type(output_data))#<class 'bytes'>

        # ret, img = cam.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the captured image into the datasets folder
        # if ret:
        #     cv2.imshow("image", img)
        # k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
        # if k == 27:
        #     break


except KeyboardInterrupt:
    print("프로그램을 종료합니다.")
finally:
    # 스트림 닫기
    stream_in.stop_stream()
    stream_in.close()
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
