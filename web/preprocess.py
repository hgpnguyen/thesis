from csv import reader, writer
from openpyxl import load_workbook
from sklearn import preprocessing
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np
from scipy.spatial import distance
from keras import backend as K
VECSIZE = 400
MAXLEN = 300

class Preprocessing:

    def __init__(self):
        self.dataset = None
        

    def get_point_data(self, filename, dataType):
        dataset = Preprocessing.load_csv(filename)
        dataset.pop(0)
        dataset = self.data_filter(dataType, dataset)
        dataset = self.point_preprocess(dataset)
        return dataset

    def get_merge_data(self, data_name, dataType='allpost'):
        post_file =   'static/data/token' + data_name + '.csv' 
        point_file =  'static/data/post_diem' + data_name + '.csv'
        post, point = Preprocessing(), Preprocessing()
        post.get_word2vec(post_file)
        point.get_point_data(point_file, dataType)
        merge_dataset = Preprocessing.merge_dict(point.dataset, post.dataset, data_name)
        self.dataset = merge_dataset
        items = merge_dataset.items()
        #print([x[1][1] for x in items])
        vector = np.array([x[1][0] for x in items])
        point_data = np.array([x[1][1] for x in items])
        target = np.array([x[1][2] for x in items])
        return vector, point_data, target
    
    def load_csv(filename):
        dataset = list()
        with open(filename, 'r', encoding="utf-8") as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue;
                dataset.append(row);
        return dataset;

    def merge_data_feature(feature_vector, point, weight=1):
        #merge_feature = feature_vector + point
        point_normal = preprocessing.normalize(point, norm='l2')
        point_weight = np.tile(point_normal, weight)
        print(point_weight.shape, feature_vector.shape)
        merge_feature = np.concatenate((feature_vector, point_weight), axis=1)
        return merge_feature
        

    def stemming(token):
        token = token.lower()
        return token

    def tokenize(self, dataset):
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
            sentences = post.split(' ')
            temp = []
            for token in sentences:
                if(len(token) > 1):
                    temp.append(stemming(token))
            #corpus.append(post)        
            corpus.append(temp)
        return corpus, target
    
    def point_preprocess(self, dataset):
        feature = []
        target = []
        keys = []
        for  i in dataset:
            temp = []
            name = Preprocessing.tranform_name(i[1])
            if name == '':
                name = i[0]
            keys.append(name)
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
        feature = np.array(feature)
        target = np.array(target)
        items = zip(feature, target)
        self.dataset = dict(zip(keys, items))
        return feature, target

    def data_filter(self, method, data):
        if method == 'nopost':
            func = lambda x: int(x[-2]) == 0
        elif method == 'post':
            func = lambda x: int(x[-2]) == 1
        else:
            func = lambda x: True
        data = list(filter(func, data))
        return data

    '''def preprocess(corpus):
        list_token = []
        for i in corpus:
            temp = list(filter(lambda x: '_' in x, i))
            list_token += temp
        list_token = set(list_token)
        check_list = list(map(lambda x: ' '.join(x.split('_')), list(list_token)))
        for i in corpus:
            for j in check_list:
                #print('token', j)
                #print('list',i)
                if j in i:
                    print(j)
        #print(check_list)'''

    def word_embed(model, word):
        return model[word] if word in model else np.zeros(VECSIZE)

    def convert_trainingdata_matrix(model, corpus):
        train_embed = np.zeros(shape=(len(corpus), MAXLEN, VECSIZE))
        for i in range(len(corpus)):
            for j in range(min(MAXLEN, len(corpus[i]))):
                train_embed[i, j] = Preprocessing.word_embed(model, corpus[i][j])
        return train_embed

    def tranform_name(name):
        name = name.split(",", 1)
        if(len(name) == 1):
            name = name[0]
            name = name.replace('_','')
            name = name.replace(' ','')
            return name
        name[1] = name[1][1:len(name[1])]
        if name[0]:
            name = name[1] + " " + name[0]
        else:
            name = name[1]
        name = name.replace('_','')
        name = name.replace(' ','')
        return name
    
    def get_word2vec(self, filename):
        dataset = Preprocessing.load_csv(filename)
        dataset = list(filter(lambda x: len(x) >= 2, dataset))
        training = {}
        word_dict = {}
        target = []
        corpus = []
        for data in dataset:
            score = data[1].replace(' ','')
            if(score != '1' and score != '0'):
                continue
            name = Preprocessing.tranform_name(data[0])
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
                    temp.append(Preprocessing.stemming(token))
            corpus.append(temp)
        #model = Word2Vec(corpus, sg=1)
        model = KeyedVectors.load_word2vec_format('static/model/word2vec.bin')
        words = list(model.wv.vocab)
        '''idf = inverse_document_frequencies(corpus)
        max_idf = max([v for k, v in idf.items()])'''
        feature = Preprocessing.convert_trainingdata_matrix(model, corpus)
        feature = np.array(feature)
        target = np.array(target)
        items = zip(feature, target)
        self.dataset = dict(zip(keys, items))
                  
        return feature, target

    def word2vec(corpus):
        model = KeyedVectors.load_word2vec_format('static/model/word2vec.bin')
        feature = Preprocessing.convert_trainingdata_matrix(model, corpus)
        feature = np.array(feature)
        return feature
        

    def merge_dict(point_dict, post_dict, data_name):
        new_dict = {}
        names_full = point_dict.keys()
        names_post = post_dict.keys()
        names_no = []
        for i in names_full:
            if i not in list(names_post):
                names_no.append(i)
        for name in names_full:
            post_feature = post_dict.get(name, (np.zeros((MAXLEN, VECSIZE)), 0))
            point_feature = point_dict[name]
            feature = (post_feature[0], point_feature[0], point_feature[1])
            new_dict[name] = feature
        if data_name == '2010':
            post_item = [(i,new_dict[i]) for i in names_post]
        else:
            merge_data = Preprocessing()
            merge_data.get_merge_data('2010', 'allpost')
            dataset = merge_data.dataset
            post_item = list(dataset.items())
            #post_item = [(i,new_dict[i]) for i in names_post]
            
        Preprocessing.fill_vector(new_dict, names_no, post_item, data_name)
        return new_dict

    def fill_vector(new_dict, names_no, post_items, data_name):
        for name in names_no:
            feature = new_dict[name]
            if data_name == '2010':
                class_item = feature[2]
                same_class_item = list(filter(lambda x: x[1][2] == class_item, post_items))
            else:
                #class_item = feature[2]
                #same_class_item = list(filter(lambda x: x[1][2] == class_item, post_items))
                same_class_item = post_items
            closest_vector = Preprocessing.get_closest_vector(feature[1], same_class_item)
            #print(closest_vector)
            new_dict[name] = (closest_vector, feature[1], feature[2])
            #print(name)

    def sparness(posts, numeric, data_train):
        for i in range(len(posts)):
            if numeric[i][-1] == 0:
                closest_vector = Preprocessing.get_closest_vector(numeric[i], data_train)
                posts[i] = closest_vector
                

    def get_closest_vector(point, item):
        other_point = [i[1][1] for i in item]
        distance_ = list(map(lambda x: distance.euclidean(point, x), other_point))
        position = 0

        min_dis = distance_[position]
        for i in range(1, len(other_point)):
            if distance_[i] < min_dis:
                position = i
                min_dis = distance_[position]
        return item[position][1][0]
        
    

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

class Metric:
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


    def msfe(y_true, y_pred):
        fpe = K.sum(K.square(y_pred * y_true - y_true))/ (K.sum(y_true) + K.epsilon())
        reverse = 1. - y_true
        fne = K.sum(K.square(y_pred * reverse - y_true * reverse))/(K.sum(reverse) + K.epsilon())
        msfe = K.square(fpe) + K.square(fne)
        return msfe

def main():
    data = '2010'
    '''infile = data + '/token' + data + '.csv'
    dataset = load_csv(infile)
    dataset = list(filter(lambda x: len(x) >= 2, dataset))
    corpus, target = tokenize(dataset)
    preprocess(corpus)'''
    merge = Preprocessing()
    vector, point, target = merge.get_merge_data(data)

if __name__ == "__main__":
   main() 
