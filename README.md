# Capstone_Design

- 2023 캡스톤 디자인 2(23/11/24 기준 정리 완료)

---

## 주요 파일 목록

- **index.py**: 메인 화면, 학습과 웹 실행 등 옵션 제공
  - **OpenCV_training.py**: 얼굴 이미지 검출 및 캡쳐 함수, 얼굴 이미지 인식 및 학습 함수
  - **flask_predict.py**: 실시간 웹캠 검열과 비속어 검열
    - **google_stream_stt.py**: 오디오 스트림을 사용하여 google api를 통한 STT 수행
    - **textCussDetect.py**: 텍스트 기반으로 문장 내에 욕설이 있는지 검출 결과 반환
- **Varable.py**: 경로와 같은 자주 변하지 않는 변수 저장

### 패치 23/11/24

<img width="398" alt="image" src="https://github.com/sleepyhood/Capstone_Design/assets/69490791/1b410d03-61ba-4f65-baa8-603cd5d718c5">

- 오디오 스트림이 get 메서드에 맞게 동작 순서 수정
- 오디오와 비디오가 웹에 정상적으로 출력되도록 수정
- 오디오에서 비속어 검출시, 부저음이 들리도록 피드백 추가
- 시연용 얼굴 이미지 yaml 제작

---

### 패치 23/11/16

<img width="294" alt="image" src="https://github.com/sleepyhood/Capstone_Design/assets/69490791/7c3a828d-85cd-4de4-8600-e91568c921e6">

- 오디오 스트림이 동작하는 동안, 읽은 문장을 바탕으로 즉각으로 욕설여부 판별
- (제작 필요) 비디오 인터페이스를 구축하여, 캠을 활성화 할 때 STT와 욕설 여부를 판별

---

### 패치 23/11/15

- 구글 API 및 서비스 > 사용자 인증 정보 > 서비스 계정 키(.json) 로컬 환경변수로 설정하여 실행가능케 함
  - set GOOGLE_APPLICATION_CREDENTIALS={경로} 로도 설정 가능
- 기존 텍스트 기반 욕설 감지 프로그램을 colab에서 vscode 환경으로 옮김
  - 단, tokenizer 객체는 최초 실행시 최소 한번은 실행 후, predict 단계로 들어가야함

---
