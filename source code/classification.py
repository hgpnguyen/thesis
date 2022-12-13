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
from preprocess import Preprocessing
import pandas as pd
import pickle
import sys
N_TIME = 30
N_ESTIMATER = 100
class Classifier():

    def __init__(self):
        self.model = None
        self.name=''

    def RandomForest(self, nes):
        clf = RandomForestClassifier(n_estimators=nes, max_features='log2')
        print(nes)
        self.model = clf
        self.name = 'Random_Forest'
        
    def SVM(self):
        clf = svm.SVC(kernel='rbf', C = 1.0)
        self.model = clf
        self.name = 'SVM'

    def C4_5(self):
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        self.model = clf
        self.name = 'C4_5'

    def train(self, features, targets):
        self.model.fit(features, targets)
        pickle.dump(self.model, open(self.name + '.sav', 'wb'))
        
        
    def test(self, features_test, target_test):
        target_pred = self.model.predict(features_test)
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
    
def model_test(model, features_train, features_test, target_train, target_test):
    acc, pre, re, f = [], [], [], []
    for i in range(N_TIME):
        model.train(features_train[0], target_train[0])
        accuracy_, precision, recall, fbeta = model.test(features_test[0], target_test[0])
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
    acc.insert(0, model.name)
    return acc

def compare_model(feature, target):
    temp = []
    a = ['']*(N_TIME+1) + ['Accuracy', 'Do lech chuan', 'Precesion', 'Recall', 'Fbeta']
    temp.append(a)
    #features_train, features_test, target_train, target_test = create_subsambling(feature, target)
    features_train, features_test, target_train, target_test = [feature[0]], [feature[1]], [target[0]], [target[1]]
    classifier = Classifier()
    classifier.RandomForest(N_ESTIMATER)
    result = model_test(classifier, features_train, features_test, target_train, target_test)
    temp.append(result)
    classifier.SVM()
    result = model_test(classifier, features_train, features_test, target_train, target_test)
    temp.append(result)
    classifier.C4_5()
    result = model_test(classifier, features_train, features_test, target_train, target_test)
    temp.append(result)
    output = []
    for i in range(len(temp[0])):
        item = [x[i] for x in temp]
        output.append(item)
    return output

            


def main():
    #Help: python ./classifiaction.py dataset dataType N_ESTIMATER
    '''data = sys.argv[1]
    dataType = sys.argv[2]
    global N_ESTIMATER
    N_ESTIMATER = int(sys.argv[3])
    file = data + '/post_diem' + data + '.csv'
    dataset = load_csv(file)
    dataset.pop(0)
    dataset = data_filter(dataType, dataset)'''
    data = sys.argv[1]
    global N_ESTIMATER
    N_ESTIMATER = int(sys.argv[1])
    trainfile = '2010/post_diem2010.csv'
    testfile = '2011/post_diem2011.csv'
    train = 'allpost'
    test = 'allpost'
    preprocess = Preprocessing()
    train_feature, train_target = preprocess.get_point_data(trainfile, train)
    test_feature, test_target = preprocess.get_point_data(testfile, test)
    feature = (train_feature, test_feature)
    target = (train_target, test_target)
    output = compare_model(feature, target)
    outfile =  'report/week12/RF_report.xlsx'
    #write_csv(outfile, output)
    label = output.pop(0)
    #sheet = data + '_' + dataType + '_' + str(N_ESTIMATER)
    sheet =   train + ' ' + test + ' ' + str(N_ESTIMATER)
    print(sheet)
    Preprocessing.write_excel(output, label, outfile, sheet)

if __name__ == "__main__":
   main()
    
