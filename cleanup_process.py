######################
## cleanup_process.py

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


def remove_hyperlink(sentence: str) -> str:
    return re.sub(http_pattern, "", str(sentence))

def replace_newline(sentence: str) -> str:
    return sentence.replace('\n', '')

def to_lower(sentence: str) -> str:
    return sentence.lower()

def remove_punctuations(sentence: str) -> str:
    return sentence.translate(str.maketrans(dict.fromkeys(string.punctuation)))

def remove_escape(sentence: str) -> str:
    return re.sub(escape_pattern, ' ', sentence)

def remove_number(sentence: str) -> str:
    return re.sub(number_pattern, '', sentence)

def remove_none_alpha(sentence: str) -> str:
    return re.sub(none_alpha_pattern, ' ', sentence)

def remove_whitespace(sentence: str):
    return re.sub(white_space_pattern, ' ', sentence.strip())

def clean_up_pipeline(sentence: str) -> str:
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuations,
                      remove_escape,
                      remove_none_alpha,
                      remove_whitespace]
    for util in cleaning_utils:
        sentence = util(sentence)
    return sentence

def remove_stop_words(words):
    result = [i for i in words if i not in ENGLISH_STOP_WORDS]
    return result

def word_stemmer(words):
    return [stemmer.stem(o) for o in words]

def word_lemmatizer(words):
    return [lemmatizer.lemmatize(o) for o in words]

def clean_token_pipeline(words):
    cleaning_utils = [remove_stop_words,word_stemmer,word_lemmatizer]
    for o in cleaning_utils:
        words = o(words)
    return words

def cleanup_process(x_train: list, x_test: list) -> (list, list):
    x_train = [clean_up_pipeline(item) for item in x_train]
    x_test = [clean_up_pipeline(item) for item in x_test]

    x_train = [word_tokenize(o) for o in x_train]
    x_test = [word_tokenize(o) for o in x_test]

    x_train = [clean_token_pipeline(o) for o in x_train]
    x_test = [clean_token_pipeline(o) for o in x_test]

    return x_train, x_test
