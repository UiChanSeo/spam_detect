#########################
# nn_predict.py
#

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def predict_rnn(model, emails, features):

    predict=[1 if o>0.5 else 0 for o in model.predict(features)]

    cnf_matrix = confusion_matrix(emails, predict)
    tn, fp, fn, tp = cnf_matrix.ravel()

    print("Precision: {:.2f}%"
          .format(100 * precision_score(emails, predict)))
    print("Recall: {:.2f}%"
          .format(100 * recall_score(emails, predict)))
    print("F1 Score: {:.2f}%"
          .format(100 * f1_score(emails, predict)))
    print("Accuracy: {:.2f}%"
          .format(100 * accuracy_score(emails, predict)))

    return predict, cnf_matrix

    
def predict_naive(model, emails, features):

    predict = model.predict(features.toarray())

    cnf_matrix = confusion_matrix(emails, predict)

    tn, fp, fn, tp = cnf_matrix.ravel()

    print("Precision: {:.2f}%"
          .format(100 * precision_score(emails, predict)))
    print("Recall: {:.2f}%"
          .format(100 * recall_score(emails, predict)))
    print("F1 Score: {:.2f}%"
          .format(100 * f1_score(emails, predict)))
    print("Accuracy: {:.2f}%"
          .format(100 * accuracy_score(emails, predict)))

    return predict, cnf_matrix
