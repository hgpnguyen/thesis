from csv import reader, writer

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

def preprocess(corpus):
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
    #print(check_list)
    

def main():
    data = '2011'
    infile = data + '/token' + data + '.csv'
    dataset = load_csv(infile)
    dataset = list(filter(lambda x: len(x) >= 2, dataset))
    corpus, target = tokenize(dataset)
    preprocess(corpus)

if __name__ == "__main__":
   main() 

