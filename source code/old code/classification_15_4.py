from csv import reader, writer
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from openpyxl import load_workbook
import pandas as pd
import sys
N_TIME = 30
N_ESTIMATER = 100


def load_csv(filename):
    dataset = list()
    with open(filename, 'r', encoding="utf-8") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue;
            dataset.append(row);
    return dataset;

def preprocessing(dataset):
    feature = []
    target = []
    for  i in dataset:
        temp = []
        for j in range(2, len(i) - 2):
            if i[j]:
                if j <= 6:
                    temp.append(float(i[j]))
                else:
                    temp.append(int(i[j]))
            else:
                temp.append(-10)
        feature.append(temp)
        target.append(int(i[- 1]))
    return np.array(feature), np.array(target)

def RandomForest(features, targets):
    clf = RandomForestClassifier(n_estimators=N_ESTIMATER, max_features='log2')
    clf.fit(features, targets)
    return clf

def SVM(features, targets):
    clf = svm.SVC(kernel='rbf', C = 1.0)
    clf.fit(features, targets)
    return clf

def C4_5(features, targets):
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(features, targets)
    return clf

def accuracy(clf,  features_test, target_test):
    target_pred = clf.predict(features_test)
    score = accuracy_score(target_test, target_pred, normalize = True)
    accuracy = score
    tn, fp, fn, tp = confusion_matrix(target_test, target_pred, labels=[0,1]).ravel()
    precision = tp + fp
    recall = tp + fn
    f1 = tp

    return accuracy, precision, recall, f1

def create_subsambling(features, targets):
    f_train, f_test, t_train, t_test = [], [], [], []
    for i in range(0, N_TIME):
        features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            targets, test_size = 0.33)
        f_train.append(features_train)
        f_test.append(features_test)
        t_train.append(target_train)
        t_test.append(target_test)
    return f_train, f_test, t_train, t_test
    
def model_test(method, features_train, features_test, target_train, target_test):
    acc, pre, re, f = [], [], [], []
    for i in range(N_TIME):
        model = method(features_train[i], target_train[i])
        accuracy_, precision, recall, fbeta = accuracy(model, features_test[i], target_test[i])
        acc.append(accuracy_)
        pre.append(precision)
        re.append(recall)
        f.append(fbeta)
    c3 = sum(pre)
    c2 = sum(re)
    c1 = sum(f)
    if c3 == 0:
        precision = 0
    else:
        precision = c1 / c3
    if c2 == 0:
        recall = 0
    else:
        recall = c1 / c2
    if precision + recall == 0:
        f_beta = 0
    else:
        f_beta = 2 * (precision * recall)/(precision + recall)    
    temp = [np.mean(acc), math.sqrt(np.var(acc)), precision, recall, f_beta]
    acc = acc + temp
    acc.insert(0, method.__name__)
    return acc

def compare_model(feature, target):
    temp = []
    a = ['']*(N_TIME+1) + ['Accuracy', 'Do lech chuan', 'Precesion', 'Recall', 'Fbeta']
    temp.append(a)
    features_train, features_test, target_train, target_test = create_subsambling(feature, target)
    result = model_test(RandomForest, features_train, features_test, target_train, target_test)
    temp.append(result)
    result = model_test(SVM, features_train, features_test, target_train, target_test)
    temp.append(result)
    result = model_test(C4_5, features_train, features_test, target_train, target_test)
    temp.append(result)
    output = []
    for i in range(len(temp[0])):
        item = [x[i] for x in temp]
        output.append(item)
    return output

def data_filter(method, data):
    if method == 'nopost':
        func = lambda x: int(x[-2]) == 0
    elif method == 'post':
        func = lambda x: int(x[-2]) == 1
    else:
        func = lambda x: True
    data = list(filter(func, data))
    return data
            
def write_csv(filename, data):
    with open(filename, 'w', newline='', encoding="utf-8") as f:
        csv_writer = writer(f)
        csv_writer.writerows(data)

def write_excel(data, label, filename, sheet):
    book = load_workbook(filename)
    df = pd.DataFrame.from_records(data, columns=label)
    writer = pd.ExcelWriter(filename, engine = 'openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet, index=0)
    writer.save()
    writer.close()

def main():
    #Help: python ./classifiaction.py dataset dataType N_ESTIMATER
    data = sys.argv[1]
    dataType = sys.argv[2]
    global N_ESTIMATER
    N_ESTIMATER = int(sys.argv[3])
    file = data + '/post_diem' + data + '.csv'
    dataset = load_csv(file)
    dataset.pop(0)
    dataset = data_filter(dataType, dataset)
    feature, target = preprocessing(dataset)
    output = compare_model(feature, target)
    outfile =  'report/week8/' + dataType + '_report.xlsx'
    #write_csv(outfile, output)
    label = output.pop(0)
    sheet = data + '_' + dataType + '_' + str(N_ESTIMATER)
    write_excel(output, label, outfile, sheet)

if __name__ == "__main__":
   main()
    
