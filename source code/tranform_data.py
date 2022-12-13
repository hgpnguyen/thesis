from csv import reader, writer
import pandas as pd
from scipy import spatial
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from openpyxl import load_workbook
VECSIZE = 400
MAXLEN = 300

def load_csv(filename):
    dataset = list()
    with open(filename, 'r', encoding="utf-8") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue;
            dataset.append(row);
    return dataset;

def stemming(token):
    token = token.lower()
    return token


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
        post = training.get(name, ['', int(data[1]), -1])
        #if len(data[3]) != 0:
        post[2] += 1
        for i in range(3, len(data)):
            #them post vao dictionary
            if(len(data[i]) !=0):
                post[0] += ' ' + data[i]
        training[name] = post
    items = list(training.items())
    keys = list(training.keys())
    #print(items[:4])
    #target = [x[1][1] for x in items]
    forum_post, target, count = [], [], []
    for x in items:
        forum_post.append(x[1][0])
        target.append(x[1][1])
        count.append(x[1][2])
    out = list(zip(keys, target, count, forum_post))
    #print(out[:4])
    '''for i in range(len(keys)):
        post = training[keys[i]][0]
        corpus.append(post)        
        #corpus.append(temp)'''
    return out

def word_embed(model, word):
    return model[word] if word in model else np.zeros(VECSIZE)

def convert_trainingdata_matrix(model, corpus):
    train_embed = np.zeros(shape=(len(corpus), MAXLEN, VECSIZE))
    for i in range(len(corpus)):
        for j in range(min(MAXLEN, len(corpus[i]))):
            train_embed[i, j] = word_embed(model, corpus[i][j])
    return train_embed

def tranform(matrix):
    return np.array([
        np.mean([x for x in sentences], axis=0) for sentences in matrix])

def word2vec(dataset):
    training = {}
    word_dict = {}
    target = []
    corpus, corpus2 = [], []
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
        sentences = post.split(' ')
        temp = []
        for token in sentences:
            if(len(token) > 1):
                temp.append(stemming(token))
        corpus.append(temp)
        corpus2.append(post)
    print(len(corpus))
    #model = Word2Vec(corpus, sg=1)
    model = KeyedVectors.load_word2vec_format('word2vec.bin')
    words = list(model.wv.vocab)
    print(len(words))
    feature = convert_trainingdata_matrix(model, corpus)
    feature = tranform(feature)

              
    return corpus2, feature, target

def nearest_3(i, feature, distance):
    point = feature[i]
    dis = list(map(lambda x: distance(x, point), feature))
    distance1 , distance2, distance3 = 10000000, 10000000, 10000000
    n1, n2, n3 = -1, -1, -1
    for x in range(len(dis)):
        if x != i:
            temp_dis = dis[x]
            if temp_dis < distance1:
                distance1 = temp_dis
                n1 = x
            elif temp_dis < distance2:
                distance2 = temp_dis
                n2 = x
            elif temp_dis < distance3:
                distance3 = temp_dis
                n3 = x
    return n1, n2, n3

def compare_feature(corpus, feature, target, distance):
    compare = []
    for i in range(len(corpus)):
        n1, n2, n3 = nearest_3(i, feature, distance)
        temp = [corpus[i], target[i], n1, target[n1], n2, target[n2], n3, target[n3]]
        compare.append(temp)
    return compare

def write_excel(data, label, filename, sheet):
    book = load_workbook(filename)
    df = pd.DataFrame.from_records(data, columns=label)
    writer = pd.ExcelWriter(filename, engine = 'openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet)
    writer.save()
    writer.close()
     
def main():
    data = '2011'
    infile = data + '/token' + data + '.csv'
    outfile = data + '/tom_tat' + data + '.csv'
    dataset = load_csv(infile)
    dataset = list(filter(lambda x: len(x) >= 2, dataset))
    item = tokenize(dataset)
    label = ['Tên SV', 'Rot', 'So bai post', 'Bài post']
    write_excel(item, label, outfile)

    #vec = list(zip(corpus, target, feature))
    #label = ['Post', 'Class', 'Vec']
    #write_excel(vec, label, outfile2, data)
    



if __name__ == "__main__":
   main() 
