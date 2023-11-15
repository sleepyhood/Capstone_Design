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


# print(os.path.dirname(__file__))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gs.main()
# stt 호출
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

# td.training() #필요시 실행
td.sentiment_predict("와", tokenizer)
td.sentiment_predict("와?", tokenizer)
"""
