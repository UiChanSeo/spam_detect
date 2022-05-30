# spam_detect
  - SPAM을 검출하는 테스트 소스 입니다.
  - 파이썬으로 작성되었읍니다.

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
  - email : 
  - notion : 