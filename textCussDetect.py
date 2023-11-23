# pip install konlpy
# pip install pyyaml h5py
# 해당 코드는 전처리와 학습 단계이므로, loaded_model을 저장했다면 별도로 실행 안함

# https://wikidocs.net/22894
# 데이터셋: https://github.com/2runo/Curse-detection-data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
import re
import os
import sys
import tensorflow
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics

from sklearn.model_selection import train_test_split

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential

# from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
tokenizer = tensorflow.keras.preprocessing.text.Tokenizer()
# 최대 길이 35라 가정시, dir 96%가 35이하의 길이를 가짐
max_len = 35
# 불용어(분석에 큰 의미를 가지지 않는 단어)를 지정해야 올바른 결과 출력
stopwords = [
    "의",
    "가",
    "이",
    "은",
    "들",
    "는",
    "좀",
    "잘",
    "걍",
    "과",
    "도",
    "를",
    "으로",
    "자",
    "에",
    "와",
    "한",
    "하다",
]

file_path = "content/textDataset.txt"


def load_and_preprocess_data():
    x = []
    y = []
    with open(file_path, "r", encoding="UTF-8") as file:
        # 파일 내용 읽기
        for line in file:
            # print(line.strip())  # 각 줄 출력, strip() 함수로 불필요한 공백 제거
            sentences = line.strip().split("|")
            x.append(sentences[0].strip())
            y.append(sentences[1].strip())

    # 분리된 문장 출력
    # 데이터프레임 생성
    df = pd.DataFrame({"X": x, "Y": y})

    # 데이터프레임 출력
    print(df)

    print("결측값 여부 :", df.isnull().values.any())

    # 0 또는 1이 아닌 행 제거
    df = df[(df["Y"] == "0") | (df["Y"] == "1")]

    # 0은 욕설아님, 1은 욕설
    print(df.groupby(df["Y"]).size().reset_index(name="count"))

    # 테스트와 시험용으로 쪼개기
    X_data = df["X"]
    y_data = df["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=0, stratify=y_data
    )

    train_data = pd.DataFrame({"document": X_train, "label": y_train})
    test_data = pd.DataFrame({"document": X_test, "label": y_test})

    print("--------훈련 데이터의 비율-----------")
    print(f"표준어 = {round(y_train.value_counts()[0]/len(y_train) * 100,3)}%")
    print(f"비속어 = {round(y_train.value_counts()[1]/len(y_train) * 100,3)}%")

    print("--------테스트 데이터의 비율-----------")
    print(f"표준어 = {round(y_test.value_counts()[0]/len(y_test) * 100,3)}%")
    print(f"비속어 = {round(y_test.value_counts()[1]/len(y_test) * 100,3)}%")

    print(train_data["document"].nunique())
    print(train_data["label"].nunique())

    train_data.drop_duplicates(subset=["document"], inplace=True)
    train_data["document"] = train_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    # null 행이 있는지 확인
    print(train_data.isnull().sum())

    train_data["document"] = train_data["document"].str.replace(
        "^ +", ""
    )  # white space 데이터를 empty value로 변경
    train_data["document"].replace("", np.nan, inplace=True)
    print(train_data.isnull().sum())
    # 공백만 있는 행이 존재함
    print(train_data.loc[train_data.document.isnull()][:5])

    # 공백만 있는 행 제거
    train_data = train_data.dropna(how="any")
    print("전처리 후 훈련용 샘플의 개수 :", len(train_data))

    test_data.drop_duplicates(
        subset=["document"], inplace=True
    )  # document 열에서 중복인 내용이 있다면 중복 제거
    test_data["document"] = test_data["document"].str.replace(
        "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", ""
    )  # 정규 표현식 수행
    test_data["document"] = test_data["document"].str.replace(
        "^ +", ""
    )  # 공백은 empty 값으로 변경
    test_data["document"].replace("", np.nan, inplace=True)  # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how="any")  # Null 값 제거
    print("전처리 후 테스트용 샘플의 개수 :", len(test_data))
    #
    # 토큰화
    X_train = []
    for sentence in tqdm(train_data["document"]):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [
            word for word in tokenized_sentence if not word in stopwords
        ]  # 불용어 제거
        X_train.append(stopwords_removed_sentence)

    X_test = []
    for sentence in tqdm(test_data["document"]):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [
            word for word in tokenized_sentence if not word in stopwords
        ]  # 불용어 제거
        X_test.append(stopwords_removed_sentence)

    # 정수 인코딩
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if value < threshold:
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print("단어 집합(vocabulary)의 크기 :", total_cnt)
    print("등장 빈도가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print("단어 집합의 크기 :", vocab_size)

    # 23/11/14 잘라진 tokenizer이 있어야 predict도 올바르게 수행
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data["label"])
    y_test = np.array(test_data["label"])

    # 다른 함수에서 사용할 수 있도록 반환
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        tokenizer,
        train_data,
        test_data,
        vocab_size,
    )


# 모델 호출
def cuss_predict(new_sentence, tok):
    try:
        loaded_model = tensorflow.keras.models.load_model("best_model.h5")
    except Exception as e:
        print(f"모델 파일이 없습니다.\n{e}")
    # loaded_model = tensorflow.keras.models.load_model("best_model.h5")

    new_sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tok.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
    print(f"new_sentence: {new_sentence}")
    score = loaded_model.predict(pad_new)  # 예측
    score = float(score[0][0])
    print(f"score: {score}")
    # score = float(loaded_model.predict(pad_new))  # 예측
    if score > 0.5:
        print("{:.2f}% 확률로 비속어가 있습니다...\n".format(score * 100))
        return True
    else:
        print("{:.2f}% 확률로 비속어가 없습니다.\n".format((1 - score) * 100))
        return False


def training():
    print("전처리 과정 진행")
    # 모델 학습
    (
        X_train,
        y_train,
        X_test,
        y_test,
        tokenizer,
        train_data,
        test_data,
        vocab_size,
    ) = load_and_preprocess_data()

    # 패딩 (길이 맞추기)
    print("문장의 최대 길이 :", max(len(review) for review in X_train))
    print("문장의 평균 길이 :", sum(map(len, X_train)) / len(X_train))
    plt.hist([len(review) for review in X_train], bins=50)
    plt.xlabel("length of samples")
    plt.ylabel("number of samples")
    # plt.show()

    def below_threshold_len(max_len, nested_list):
        count = 0
        for sentence in nested_list:
            if len(sentence) <= max_len:
                count = count + 1
        print(
            "전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s"
            % (max_len, (count / len(nested_list)) * 100)
        )

    # 최대 길이 35라 가정시, dir 96%가 35이하의 길이를 가짐
    # max_len = 35
    print(below_threshold_len(max_len, X_train))

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(X_train.dtype)
    print(y_train.dtype)  # 종속변수가 object 타입으로 나온다;;; 삽질만 12번했다.
    X_train = X_train.astype("int32")
    y_train = y_train.astype("int32")
    X_test = X_test.astype("int32")
    y_test = y_test.astype("int32")
    print(y_train.dtype)

    # 모든 요소가 숫자형인지 확인
    print("\ncheck\n")
    notNumX = 0
    notNumY = 0
    for element in X_train:
        if not np.issubdtype(type(element), np.number):
            notNumX += 1
            # print(f"X_train: 숫자가 아닌 값이 발견되었습니다: {element}")

    for element in y_train:
        if not np.issubdtype(type(element), np.number):
            notNumY += 1
            # print(f"y_train: 숫자가 아닌 값이 발견되었습니다: {element}")

    print(f"notNumX: {notNumX} / notNumY: {notNumY}")
    print(X_train[0])

    ##########################
    # LSTM으로 문장 분류

    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation="sigmoid"))

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
    mc = ModelCheckpoint(
        "best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
    )

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    history = model.fit(
        X_train,
        y_train,
        epochs=7,
        callbacks=[es, mc],
        batch_size=64,
        validation_split=0.2,
    )

    model.save("best_model.h5")
    # model_encoded = model.encode("cp949")

    loaded_model = tensorflow.keras.models.load_model("best_model.h5")
    loaded_model.summary()

    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
    return tokenizer


# (
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     tokenizer,
#     train_data,
#     test_data,
# ) = load_and_preprocess_data()
# # training()
# sentiment_predict("와", tokenizer)
