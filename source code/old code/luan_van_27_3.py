#!/usr/bin/env python
# -*- coding: utf-8 -*-
from csv import reader, writer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dropout
from keras.layers import LSTM
from keras.optimizers import SGD
from keras import backend as K
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import math
#theano.config.optimizer="None"
N_TIME = 100
VECSIZE = 400
MAXLEN = 300
word2weight = None

def load_csv(filename):
    dataset = list()
    with open(filename, 'r', encoding="utf-8") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue;
            dataset.append(row);
    return dataset;

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c1 == 0 or c2 == 0 or c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / (c2 + K.epsilon())

    # How many relevant items are selected?
    recall = c1 / (c3 + K.epsilon())

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    #return f1_score
    return c1


def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.


    # How many selected items are relevant?
    precision = c1 / (c2 + K.epsilon())

    #return precision
    return c2


def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))


    recall = c1 / (c3 + K.epsilon())

    #return recall
    return c3

def mfe(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(K.eval(y_true), K.eval(y_pred), labels=[0,1]).ravel()
    fpe = fn / (tp + fn)
    fne = fp / (fp + tn)
    mfe = fpe + fne
    return mfe

def msfe(y_true, y_pred):
    fpe = K.sum(K.square(y_pred * y_true - y_true))/ (K.sum(y_true) + K.epsilon())
    print(K.sum(y_true))
    reverse = 1. - y_true
    print(K.sum(reverse))
    fne = K.sum(K.square(y_pred * reverse - y_true * reverse))/(K.sum(reverse) + K.epsilon())
    msfe = fpe + fne
    #y_true = K.eval(y_true)
    #y_pred = K.eval(y_pred)
    '''tn, fp, fn, tp = confusion_matrix(K.eval(y_true), K.eval(y_pred), labels=[0,1]).ravel()
    fpe = fn / (tp + fn)
    fne = fp / (fp + tn)
    msfe = fpe * fpe + fne * fne'''
    return msfe
    

def word_embed(model, word):
    return model[word] if word in model else np.zeros(VECSIZE)

def tranform(model, corpus):
    idf = inverse_document_frequencies(corpus)
    max_idf = max([v for k, v in idf.items()])
    return np.array([
        np.mean([word_embed(model, x) * idf.get(x, max_idf) for x in sentences if sentences] or [np.zeros(VECSIZE)], axis=0) for sentences in corpus])

def tranform(matrix):
    return np.array([
        np.mean([x for x in sentences], axis=0) for sentences in matrix])

def topic_to_matrix(model, sentences):
    matrix = np.zeros((MAXLEN, VECSIZE))
    for i in range(min(MAXLEN, len(sentences))):
        matrix[i] = word_embed(model, sentences[i])
    return matrix

def convert_trainingdata_matrix(model, corpus):
    train_embed = np.zeros(shape=(len(corpus), MAXLEN, VECSIZE))
    for i in range(len(corpus)):
        for j in range(min(MAXLEN, len(corpus[i]))):
            train_embed[i, j] = word_embed(model, corpus[i][j])
    return train_embed

def inverse_document_frequencies(corpus):
    idf_value = {}
    global word2weight
    all_token = set([word for sentence in corpus for word in sentence])
    for token in all_token:
        contain_token = map(lambda doc: token in doc, corpus)
        idf_value[token] =  math.log(len(corpus)/(sum(contain_token)))
    word2weight = idf_value
    return idf_value

def term_frequency(term, doc):
    count = doc.count(term)
    max_tf_ = max_tf(doc)
    return count/max_tf_

def max_tf(doc):
    count = [doc.count(t) for t in doc]
    if not count:
        return 1
    return max(count)
    
def stemming(token):
    token = token.lower()
    return token

def print_some(item, n):
    for i in range(n):
        print(item[i])

def naive_bayers(features, targets):
    clf = GaussianNB()
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

def CNN(features, targets, word2vec):
    depth, height, width = 1, len(targets), len(features[1]);
    model = Sequential()
    if not word2vec:
        model.add(Embedding(1000, VECSIZE, input_length=width))
        model.add(Conv1D(128, 20 , padding="same"))
    else:
        model.add(Conv1D(128, 20 , padding="same", input_shape=(MAXLEN, VECSIZE)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(64, 10 , padding="same"))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01)
    model.compile(loss=msfe, optimizer=opt,
	metrics=["accuracy", precision, recall, f1_score])
    model.fit(features, targets, batch_size=64, epochs=30)
    return model

def DeepNN(data, label):
    width = len(data[0])

    input_s = (width,)
    model = Sequential()
    model.add(Dense(3000, activation='sigmoid', input_shape=input_s))
    model.add(Dense(1000, activation='sigmoid'))
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.05)
    model.compile(loss=msfe, optimizer=opt,
	metrics=["accuracy", precision, recall, f1_score])
    model.fit(data, label, batch_size=64, epochs=30)
    return model

def Lstm(features, targets, word2vec):
    depth, height, width = 1, len(targets), len(features[1]);
    model = Sequential()
    if not word2vec:
        model.add(Embedding(1000, VECSIZE, input_length=width))
        model.add(Conv1D(128, 10 , padding="same"))
    else:
        model.add(Conv1D(128, 10 , padding="same", input_shape=(MAXLEN, VECSIZE)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01)
    model.compile(loss=msfe, optimizer=opt,
	metrics=["accuracy", precision, recall, f1_score])
    model.fit(features, targets, batch_size=64, epochs=30)
    return model

def accuracy(clf,  features_test, target_test):
    target_pred = clf.predict(features_test)
    score = accuracy_score(target_test, target_pred, normalize = True)
    #print("Predict", target_pred)
    #print("Result", target_test)
    accuracy = score
    tn, fp, fn, tp = confusion_matrix(target_test, target_pred, labels=[0,1]).ravel()
    precision = tp + fp
    recall = tp + fn
    f1 = tp
    
    return accuracy, precision, recall, f1

def accuracyCNN(clf,  features_test, target_test):
    score = clf.evaluate(features_test, target_test, verbose=0)
    pred = clf.predict_classes(np.array(features_test))
    target_pred = []
    for i in pred:
        target_pred.append(i[0])
    #print("Predict", target_pred)
    #print("Result", target_test)
    accuracy = score[1]
    precision = score[2]
    recall = score[3]
    f1 = score[4]
    if math.isnan(f1):
        f1 = 0
    #mean = np.mean(accuracy)
    #do_lech = math.sqrt(np.var(accuracy))
    #accuracy.append(mean)
    #accuracy.append(do_lech)
    #return accuracy, precision, recall, f1, target_pred
    return accuracy, precision, recall, f1
    
def tokenize(dataset):
    training = {}
    word_dict = {}
    target = []
    corpus = []
    for data in dataset:
        score = data[1].replace(' ','')
        if(score != '1' and score != '0'):
            continue
        name = data[0].replace(' ', '')
        post = training.get(name, ['', int(data[1])])
        for i in range(3, len(data)):
            #them post vao dictionary
            post[0] += ' ' + data[i]
        training[name] = post
    items = list(training.items())
    keys = list(training.keys())
    target = [x[1][1] for x in items]
    for i in range(len(keys)):
        post = training[keys[i]][0]
        '''sentences = post[0].split(' ')
        temp = []
        for token in sentences:
            if(len(token) > 1):
                temp.append(stemming(token))'''
        corpus.append(post)        
        #corpus.append(temp)
    return corpus, target

def get_vector(dataset):
    training = {}
    word_dict = {}
    target = []
    feature = []
    for data in dataset:
        score = data[1].replace(' ','')
        if(score != '1' and score != '0'):
            continue
        name = data[0].replace(' ', '')
        post = training.get(name, ['', int(data[1])])
        for i in range(3, len(data)):
            #them post vao dictionary
            post[0] += ' ' + data[i]
        training[name] = post
    items = list(training.items())
    keys = list(training.keys())
    target = [x[1][1] for x in items]
    for i in range(len(keys)):
        post = training[keys[i]]
        #dictionary
        words = post[0].split(' ')
        for token in words:
            if(len(token) > 1):
                token = stemming(token)
                word_ls = word_dict.get(token, [0 for i in range(len(keys))])
                word_ls[i] = 1
                word_dict[token] = word_ls
    witems = list(word_dict.items())
    wkeys = list(word_dict.keys())
    for i in range(len(keys)):
        temp = [x[1][i] for x in witems]
        feature.append(temp)
    print(len(items))
    print(len(feature[1]))
    return (feature, target)

def word2vec(dataset):
    training = {}
    word_dict = {}
    target = []
    corpus = []
    for data in dataset:
        score = data[1].replace(' ','')
        if(score != '1' and score != '0'):
            continue
        name = data[0].replace(' ', '')
        post = training.get(name, ['', int(data[1])])
        for i in range(3, len(data)):
            #them post vao dictionary
            post[0] += ' ' + data[i]
        training[name] = post
    items = list(training.items())
    keys = list(training.keys())
    target = [x[1][1] for x in items]
    for i in range(len(keys)):
        post = training[keys[i]]
        sentences = post[0].split(' ')
        temp = []
        for token in sentences:
            if(len(token) > 1):
                temp.append(stemming(token))
        corpus.append(temp)
    print(len(corpus))
    #model = Word2Vec(corpus, sg=1)
    model = KeyedVectors.load_word2vec_format('word2vec.bin')
    words = list(model.wv.vocab)
    print(len(words))
    idf = inverse_document_frequencies(corpus)
    max_idf = max([v for k, v in idf.items()])
    '''vec = np.array([
        np.mean([word_embed(model, x) * idf.get(x, max_idf) for x in sentences if sentences] or [np.zeros(VECSIZE)], axis=0) for sentences in corpus])
    print(vec)'''
    feature = convert_trainingdata_matrix(model, corpus)
    #feature2 = tranform(model, corpus)
    #vec2 = tranform(feature)
    #print(vec2)
    '''if(vec == vec2):
        print("OK")
    else:
        print("notOK")'''
              
    return (feature, target)


def tf_idf(dataset):
    corpus, target = tokenize(dataset)
    tokenize2 = lambda doc: doc.lower().split(" ")
    #print(corpus[:10])
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=3, tokenizer=tokenize2)
    sklearn_representation = sklearn_tfidf.fit_transform(corpus)
    feature = sklearn_representation.toarray()
    for i in range(N_TIME):
        features_train, features_test, target_train, target_test, corpus_train, corpus_test = train_test_split(feature,
                                                                            target, corpus, test_size = 0.33)
        train = list(zip(target_train, features_train))
        test = list(zip(target_test, corpus_test, features_test))
        filename = 'test/' + str(i) + '.csv'
        trainfile = 'train/'  + str(i) + '.csv'
        write_csv2(test, filename)
        write_csv2(train, trainfile)
    '''idf = inverse_document_frequencies(corpus)
    item = idf.keys()
    tfidf_docs = []
    for sentence in corpus:
        print('a')
        doc_tfidf = []
        for term in item:
            tf = term_frequency(term, sentence)
            doc_tfidf.append(tf * idf[term])
        tfidf_docs.append(doc_tfidf)'''
    return (feature, target)
def str_to_array(feature):
    feature = feature[1:len(feature)-1]
    feature = feature.split(' ')
    feature = list(filter(lambda x: len(x) != 0, feature))
    feature = list(map(lambda x: x.replace('\n', ''), feature))
    feature = list(map(lambda x: float(x), feature))
    return feature

def test_model(name, method, deepN, word2vec):
    model = None
    acc, pre, re, f = [], [], [], []
    for i in range(N_TIME):
        testfile = 'test/' + str(i) + '.csv'
        trainfile = 'train/' + str(i) + '.csv'
        test = load_csv(testfile)
        train = load_csv(trainfile)
        features_test = np.array([str_to_array(x[2]) for x in test])
        corpus = [x[1] for x in test]
        target_test = np.array([int(x[0]) for x in test])
        features_train = np.array([str_to_array(x[1]) for x in train])
        target_train = np.array([int(x[0]) for x in train])
        if deepN:
            if i == 0:
                model = method(features_train, target_train, word2vec)
            accuracy_, precision, recall, fbeta, predict = accuracyCNN(model, features_test, target_test)
        else:
            model = method(features_train, target_train)
            accuracy_, precision, recall, fbeta, predict = accuracy(model, features_test, target_test)
        result = list(zip(predict, target_test, corpus, features_test))
        filename = 'result/' + str(method.__name__) + '/' + str(method.__name__) + '_' + str(i) + '.csv'
        write_csv2(result, filename)
        acc.append(accuracy_)
        pre.append(precision)
        re.append(recall)
        f.append(fbeta)
    temp = [np.mean(acc), math.sqrt(np.var(acc)), np.mean(pre), math.sqrt(np.var(pre)),
            np.mean(re), math.sqrt(np.var(re)), np.mean(f), math.sqrt(np.var(f))]
    for j in temp:
        acc.append(j)
    acc.insert(0, name)
    return acc
    

def compare_model(feature, target, method):
    word2vec = False
    if(method == 1):
        temp = ['LSTM_Binary', 'SVM_Binary', 'C4.5_Binary', 'CNN_Binary']
    elif(method == 2):
        temp = ['LSTM_Idf', 'SVM_Idf', 'C4.5_Idf', 'CNN_Idf']
    else:
        temp = ['LSTM_vec', 'SVM_vec', 'C4.5_vec', 'CNN_vec']
        word2vec = True
    acc1, acc2, acc3, acc4 = [], [], [], []
    pre1, pre2, pre3, pre4 = [], [], [], []
    re1, re2, re3, re4 = [], [], [], []
    fbeta1, fbeta2, fbeta3, fbeta4 = [], [], [], []
    features_train, features_test, target_train, target_test = train_test_split(feature,
                                                                            target, test_size = 0.33)
    model = CNN(features_train, target_train, word2vec)
    lstm_model = Lstm(features_train, target_train, word2vec)
    for i in range(N_TIME):
        
        
        print("CNN")
        accuracy4, precision4, recall_4, f1_4 = accuracyCNN(model, features_test, target_test)
        #accuracy4, precision4, recall_4, f1_4 = 0,0,0,0
        acc4.append(accuracy4)
        pre4.append(precision4)
        re4.append(recall_4)
        fbeta4.append(f1_4)
        print("LSTM")
        accuracy1, precision1, recall_1, f1_1 = accuracyCNN(lstm_model, features_test, target_test)
        #accuracy1, precision1, recall_1, f1_1 = 0,0 ,0,0
        acc1.append(accuracy1)
        pre1.append(precision1)
        re1.append(recall_1)
        fbeta1.append(f1_1)
        if word2vec:
            features_train = tranform(features_train)
            features_test = tranform(features_test)
        
        print("SVM")
        clf = SVM(features_train, target_train)
        accuracy2, precision2, recall_2, f1_2 = accuracy(clf, features_test, target_test)
        acc2.append(accuracy2)
        pre2.append(precision2)
        re2.append(recall_2)
        fbeta2.append(f1_2)
        print("C4_5")
        clf = C4_5(features_train, target_train)
        accuracy3, precision2, recall_3, f1_3 = accuracy(clf, features_test, target_test)
        acc3.append(accuracy3)
        pre3.append(precision2)
        re3.append(recall_3)
        fbeta3.append(f1_3)
        features_train, features_test, target_train, target_test = train_test_split(feature,
                                                                            target, test_size = 0.33)
        #if(accuracy2 == accuracy4):
            #print(accuracy2)
    
    output = []
    output.append(acc1)
    output.append(acc2)
    output.append(acc3)
    output.append(acc4)
    c3 = list(map(lambda x: sum(x), [pre1, pre2, pre3, pre4])) #TP + FP
    c2 = list(map(lambda x: sum(x), [re1, re2, re3, re4]))  #TP + FN
    c1 = list(map(lambda x: sum(x), [fbeta1, fbeta2, fbeta3, fbeta4]))    #True positive
    pre_mean, re_mean, fbeta_mean = [], [], []
    
    for i in range(len(output)):
        l = output[i]
        mean = np.mean(l)
        do_lech = math.sqrt(np.var(l))
        l.append(mean)
        l.append(do_lech)
        l.insert(0, temp[i])
        if c3[i] != 0:
            precision = c1[i] / c3[i]
        else:
            precision = 0
        l.append(precision)
        if c2[i] != 0:
            recall = c1[i] / c2[i]
        else:
            recall = 0
        l.append(recall)
        f_beta = 2 * (precision * recall)/(precision + recall)
        if math.isnan(f_beta):
            f_beta = 0
        l.append(f_beta)
    '''for i in range(0, N_TIME):
        item = ['', acc1[i], acc2[i], acc3[i], acc4[i]]
        output.append(item)
    output.append(['Mean', sum(acc1)/len(acc1), sum(acc2)/len(acc2), sum(acc3)/len(acc3), sum(acc4)/len(acc4)])
    output.append(['Do lech chuan', math.sqrt(np.var(acc1)), math.sqrt(np.var(acc2)), math.sqrt(np.var(acc3)), math.sqrt(np.var(acc4))])'''
    return output

def compare_model2(method):
    temp = ['Naive Bayer_' + method, 'SVM_' + method, 'C4.5_' + method, 'CNN_' + method]
    word2vec = False
    if method == 'vec':
        word2vec = True
    output = []
    result = test_model(temp[0], naive_bayers, False, word2vec)
    output.append(result)

    result = test_model(temp[1], SVM, False, word2vec)
    output.append(result)

    result = test_model(temp[2], C4_5, False, word2vec)
    output.append(result)

    result = test_model(temp[3], CNN, True, word2vec)
    output.append(result)

    return output
'''def compare_model(feature, target, method):
    word2vec = False
    if method == 1:
        met = 'Binary'
    elif method == 2:
        met = 'Idf'
    else:
        met = 'vec'
        word2vec = True
        feature = tranform(feature)

    temp = 'DeepNN_' + met
    features_train, features_test, target_train, target_test = train_test_split(feature,
                                                                            target, test_size = 0.33)
    model = DeepNN(features_train, target_train)
    acc, pre, re, fbeta = [], [], [], []
    output = []
    for i in range(N_TIME):
        features_train, features_test, target_train, target_test = train_test_split(feature,
                                                                            target, test_size = 0.33)
        accuracy1, precision1, recall_1, f1_1 = accuracyCNN(model, features_test, target_test)
        acc.append(accuracy1)
        pre.append(precision1)
        re.append(recall_1)
        fbeta.append(f1_1)
    mean = np.mean(acc)
    do_lech = math.sqrt(np.var(acc))
    acc.append(mean)
    acc.append(do_lech)
    precision = sum(fbeta)/sum(pre)
    if math.isnan(precision):
        precision = 0
    recall = sum(fbeta)/sum(re)
    f_beta = 2 * (precision * recall)/(precision + recall)
    if math.isnan(f_beta):
        f_beta = 0
    acc.append(precision)
    acc.append(recall)
    acc.append(f_beta)
    acc.insert(0, temp)
    output.append(acc)
    return output'''
    
    
    


def write_csv(result):
    last = []
    for i in range(len(result[0])):
        for x in result:
            last.append(x[i])
    firstRow = ['']*(N_TIME + 1)
    firstRow.append('Accuracy Mean')
    firstRow.append('Do lech chuan')
    firstRow.append('Precision Mean')
    firstRow.append('Recall Mean')
    firstRow.append('F1 Mean')
        
    with open('report.csv', 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(firstRow)
        csv_writer.writerows(last)

def write_csv2(result, filename):
    with open(filename, 'w', newline='', encoding="utf-8") as f:
        csv_writer = writer(f)
        csv_writer.writerows(result)

def phan_tich(method1, method2):
    for i in range(N_TIME):
        first = 'result/' + method1 + '/' + method1 + '_' + str(i) + '.csv'
        second = 'result/' + method2 + '/' + method2 + '_' + str(i) + '.csv'
        temp1 = load_csv(first)
        predict1 = [int(x[0]) for x in temp1]
        temp2 = load_csv(second)
        predict2 = [int(x[0]) for x in temp2]
        target = [int(x[1]) for x in temp2]
        corpus = [x[2] for x in temp2]
        feature = [x[3] for x in temp2]
        #analys = list(zip(predict1, predict2, target, corpus))
        filename = 'result/phan_tich/' + str(i) + '.csv'
        CNN_C4_5, positive, negative = [['C4_5 predict', 'CNN_predict', 'answer']], [['C4_5 predict', 'CNN_predict', 'answer']], [['C4_5 predict', 'CNN_predict', 'answer']]
        for i in range(len(temp1)):
            if predict1[i] == 1 and predict2[i] == 0 and target[i] == 0:
                CNN_C4_5.append([1, 0, 0, corpus[i], feature[i]])                
            if predict1[i] == 0:
                negative.append([0, ' ', target[i], corpus[i], feature[i]])
            if predict1[i] == 1:
                positive.append([1, ' ', target[i], corpus[i], feature[i]])
        CNN_C4_5.append([])
        negative.append([])
        positive.append([])
        analys = CNN_C4_5 + negative + positive
        write_csv2(analys, filename)
 
def main():
    dataset = load_csv('2010/demo2010.csv')
    dataset = list(filter(lambda x: len(x) >= 2, dataset))
    #print(dataset[:20])
    result = []

    output = word2vec(dataset)
    feature = output[0]
    target = np.array(output[1])
    result.append(compare_model(feature, target, 3))
    write_csv(result)
    output = get_vector(dataset)
    feature = np.array(output[0])
    target = np.array(output[1])
    result.append(compare_model(feature, target, 1))
    write_csv(result)
    #output = tf_idf(dataset)
    '''result.append(compare_model2('idf'))
    write_csv(result)
    phan_tich('C4_5', 'CNN')'''
    feature = np.array(output[0])
    target = np.array(output[1])
    result.append(compare_model(feature, target, 2))

    write_csv(result)

if __name__ == "__main__":
   main()
    
