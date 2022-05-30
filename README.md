# spam_detect
  - SPAM을 검출하는 테스트 소스입니다.
  - 이 소스는 "나이브 베이즈"와 "RNN-LSTM"을 이용하여 스팸을 검출합니다.
  - 파이썬으로 작성되었습니다.

# 사전 설치
  - pip install -r requirement.txt

# 사용법
  - python main.py [options]
  - options:
    - -h, --help      show this help message and exit
    - --batch_size=Number
    - --epochs=Number
    - --test_class LSTMRNN / TBC / TBV
  ----
    - TBC : 나이브 베이즈 with CountVectorizer 테스트
    - TBV : 나이브 베이즈 with TfidfVectorizer 테스트
    - LSTM : RNN-LSTM 테스트
  ----

# 작성자

  - 서의찬
  - email : seouichan@naver.com
  - notion : https://www.notion.so/ML-67fb5afa8a304e089ac552e421198415
