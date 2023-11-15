# Capstone_Design
* 2023 캡디2 작업중인것
---
## 주요 파일 목록
* index.py: 모듈과 웹 화면을 연결하여 호출하는 화면
* google_stream_stt.py: 오디오 스트림을 사용하여 google api를 통한 STT 수행
* opencv_predict.py: 웹캠을 사용하여 실시간 특정인 학습 및 기타 인원 모자이크 적용
* textCussDetect: 텍스트 기반으로 문장 내에 욕설이 있는지 검출

### 패치 23/11/15
* 구글 API 및 서비스 > 사용자 인증 정보 > 서비스 계정 키(.json) 로컬 환경변수로 설정하여 실행가능케 함
  * set GOOGLE_APPLICATION_CREDENTIALS={경로} 로도 설정 가능
* 기존 텍스트 기반 욕설 감지 프로그램을 colab에서 vscode 환경으로 옮김
  * 단, tokenizer 객체는 최초 실행시 최소 한번은 실행 후, predict 단계로 들어가야함
---
