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
) = td.load_and_preprocess_data()
"""

# td.training() #필요시 실행
# td.sentiment_predict("안녕하세요", tokenizer)


def googleSTT():
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
        while not stream.closed:
            # if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
            #     print("프로그램을 종료합니다.")
            #     break
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()

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
            td.sentiment_predict(sttResult, tokenizer)
            #

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


def main():
    while True:
        # q 또는 esc 키를 누르면 반복문 종료
        if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
            print("프로그램을 종료합니다.")
            break
        # with mic_manager as stream:

        # 여기에서 수행할 작업을 추가


# if __name__ == "__main__":
#     main()


# stt 호출
googleSTT()
