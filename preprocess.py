######################
## preprocess.py

import re
import string

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

http_pattern=r"http\S+"
escape_pattern=r'[\n\t\r\n\,]'
none_alpha_pattern=r'[^a-zA-Z]'
white_space_pattern=r'[ ]+'
number_pattern=r'\d+'

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


class CleanEmail:
    @staticmethod
    def remove_hyperlink(sentence: str) -> str:
        return re.sub(http_pattern, "", str(sentence))

    @staticmethod
    def replace_newline(sentence: str) -> str:
        return sentence.replace('\n', '')

    @staticmethod
    def to_lower(sentence: str) -> str:
        return sentence.lower()

    @staticmethod
    def remove_punctuations(sentence: str) -> str:
        return sentence.translate(str.maketrans(dict.fromkeys(string.punctuation)))

    @staticmethod
    def remove_escape(sentence: str) -> str:
        return re.sub(escape_pattern, ' ', sentence)

    @staticmethod
    def remove_number(sentence: str) -> str:
        return re.sub(number_pattern, '', sentence)

    @staticmethod
    def remove_none_alpha(sentence: str) -> str:
        return re.sub(none_alpha_pattern, ' ', sentence)

    @staticmethod
    def remove_whitespace(sentence: str) -> str:
        return re.sub(white_space_pattern, ' ', sentence.strip())

    @staticmethod
    def remove_stop_words(words: list) -> list:
        result = [i for i in words if i not in ENGLISH_STOP_WORDS]
        return result

    @staticmethod
    def word_stemmer(words: list) -> list:
        return [stemmer.stem(o) for o in words]

    @staticmethod
    def word_lemmatizer(words: list) -> list:
        return [lemmatizer.lemmatize(o) for o in words]


preprocesss = [CleanEmail.remove_hyperlink,
               CleanEmail.replace_newline,
               CleanEmail.to_lower,
               CleanEmail.remove_number,
               CleanEmail.remove_punctuations,
               CleanEmail.remove_escape,
               CleanEmail.remove_none_alpha,
               CleanEmail.remove_whitespace,
               word_tokenize,
               CleanEmail.remove_stop_words,
               CleanEmail.word_stemmer,
               CleanEmail.word_lemmatizer]


def pipelines(sentence: str) -> str:
    for process in preprocesss:
        sentence = process(sentence)
    return sentence


def preprocess(x_train: list, x_test: list) -> (list, list):
    x_train = [pipelines(item) for item in x_train]
    x_test = [pipelines(item) for item in x_test]

    return x_train, x_test
