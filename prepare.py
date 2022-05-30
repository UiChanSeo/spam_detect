######################
# prepare.py
#

from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
import email

path = './'

easy_ham_paths = glob.glob(path+'easy_ham/*')
easy_ham_2_paths = glob.glob(path+'easy_ham_2/*')
hard_ham_paths = glob.glob(path+'hard_ham/*')
spam_paths = glob.glob(path+'spam/*')
spam_2_paths = glob.glob(path+'spam_2/*')

ham_path = [
    easy_ham_paths,
    easy_ham_2_paths,
    hard_ham_paths ]

spam_path = [spam_paths, spam_2_paths]

###############################
# read email and append to list
#
def get_email_content(email_path):
    with open(email_path, encoding='latin1') as file:
        try:
            msg = email.message_from_file(file)
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload() # prints the raw text
        except Exception as e:
            print(e)

def get_email_contents(email_paths):
    return [get_email_content(o) for o in email_paths]


##########################
# 빈 데이터를 제거
#
def remove_null(datas,labels):
    not_null_idx = [i for i,o in enumerate(datas) if o is not None]
    return np.array(datas)[not_null_idx],np.array(labels)[not_null_idx]

##########################
# prepare data
#
def prepare_data():
    # ham 데이터 생성
    ham_sample = np.array([train_test_split(o) for o in ham_path])

    ham_train = np.array([])
    ham_test = np.array([])
    for o in ham_sample:
        ham_train = np.concatenate((ham_train,o[0]),axis=0)
        ham_test = np.concatenate((ham_test,o[1]),axis=0)

    # spam 데이터 생성
    spam_sample=np.array([train_test_split(o) for o in spam_path])

    spam_train = np.array([])
    spam_test = np.array([])

    for o in spam_sample:
        spam_train = np.concatenate((spam_train,o[0]),axis=0)
        spam_test = np.concatenate((spam_test,o[1]),axis=0)

    # train 데이터와 라벨 생성
    x_train = np.concatenate((ham_train,spam_train))
    y_train = np.concatenate(([0]*ham_train.shape[0], [1]*spam_train.shape[0]))

    # test 데이터와 라벨 생성
    x_test = np.concatenate((ham_test,spam_test))
    y_test = np.concatenate(([0]*ham_test.shape[0], [1]*spam_test.shape[0]))

    # 데이터 썩기 (ham과 spam이 따로 나뉘지 않고 섞이도록 함.
    # 데이터를 섞는 정도에 따라서 Recall과 Precision에 변동이 있다.
    train_shuffle_index = np.random.permutation(np.arange(0,x_train.shape[0]))
    test_shuffle_index = np.random.permutation(np.arange(0,x_test.shape[0]))
    x_train = x_train[train_shuffle_index]
    y_train = y_train[train_shuffle_index]
    x_test = x_test[test_shuffle_index]
    y_test = y_test[test_shuffle_index]

    x_train = get_email_contents(x_train)
    x_test = get_email_contents(x_test)

    x_train,y_train = remove_null(x_train,y_train)
    x_test,y_test = remove_null(x_test,y_test)

    return (x_train, y_train, x_test, y_test)

