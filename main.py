from prepare import prepare_data
from cleanup_process import cleanup_process
from process import process
from train import train_simplernn, train_lstm
from train import train_naive_with_count
from train import train_naive_with_tfidf
from predict import predict_rnn, predict_naive

from plot import draw_confusion_matrix
import numpy as np
import argparse

def parse_args() -> (int, int):
    parser = argparse.ArgumentParser(description='SPAM Detect')

    parser.add_argument('--batch_size',
                        type=int, default=512)
    parser.add_argument('--epochs',
                        type=int, default=20)
    parser.add_argument('--test_class',
                        type=str, default='NN',
                        help="SIMPLERNN / LSTMRNN / TBC / TBV")
    args = parser.parse_args()

    return args.batch_size, args.epochs, args.test_class

trains = {"SIMPLERNN": train_simplernn,
          "LSTMRNN": train_lstm,
          "TBC": train_naive_with_count,
          "TBV": train_naive_with_tfidf}

predicts = {"SIMPLERNN": predict_rnn,
            "LSTMRNN": predict_rnn,
            "TBC": predict_naive,
            "TBV": predict_naive}


if __name__=="__main__":

    batch_size, epochs, class_name = parse_args()
    print(f'----start----')
    print(f'')
    print(f'batch_size = {batch_size}')
    print(f'epochs     = {epochs}')
    print(f'')

    # data prepare
    x_train, y_train, x_test, y_test = prepare_data()

    # data process
    x_train, x_test = cleanup_process(x_train, x_test)

    model, features = trains[class_name](x_train, x_test,
                                         y_train, y_test,
                                         is_model_save = True,
                                         epochs = epochs,
                                         batch_size = batch_size)

    predict, cnf_matrix = predicts[class_name](model, y_test, features)
    
    draw_confusion_matrix(cnf_matrix, classes=['NoneSpam', 'Spam'],
                          normalize=False, title='Confusion Matrix') 
