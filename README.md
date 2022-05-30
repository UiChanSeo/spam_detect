# spam_detect
  - SPAM을 검출하는 테스트 소스입니다.
  - 이 소스는 "나이브 베이즈"와 "RNN-LSTM"을 이용하여 스팸을 검출합니다.
  - 파이썬으로 작성되었습니다.

# 사전 설치 및 정보
  - 파이썬 버젼 및 환경 :
    - 파이썬 버젼 : 3.10.0
    - 환경 : virtualenv
  - 추가 모듈
    - 추가 모듈은 requirement.txt에 있습니다.
    - 설치는 다음과 같이 합니다.
      * pip install -r requirement.txt

# 사용법
  - python main.py [options]
  - options:
    - -h, --help      show this help message and exit
    - --batch_size=Number (지정하지 않을 경우 기본 512)
    - --epochs=Number (지정하지 않을 경우 기본 20)
    - --test_class LSTMRNN / TBC / TBV
  ----
    - test_class 설명 : 
      - TBC : 나이브 베이즈 with CountVectorizer 테스트
      - TBV : 나이브 베이즈 with TfidfVectorizer 테스트
      - LSTM : RNN-LSTM 테스트
    
    - 예시 : 
      - 나이브 베이즈 : python main.py --test_class=TBC
      - RNN-LSTM, epochs=10 : python main.py --test_class=LSTMRNN --epochs=10
  ----

# 작성자

  - 서의찬
  - email : seouichan@naver.com
  - notion : https://www.notion.so/ML-67fb5afa8a304e089ac552e421198415
