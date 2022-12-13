from gensim.models import Word2Vec
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
def createCorpus(data, data2):
    corpus = []
    for i in data:
        for j in range(3, len(i)):
            temp = i[j].split(' ')
            temp = list(filter(lambda x: len(x)> 1, temp))
            corpus.append(temp)
    for i in data2:
        for j in range(3, len(i)):
            temp = i[j].split(' ')
            temp = list(filter(lambda x: len(x)> 1, temp))
            corpus.append(temp)
    return corpus
    

def main():
    data2010 = load_csv('2010/token2010.csv')
    data2011 = load_csv('2011/token2011.csv')
    corpus = createCorpus(data2010, data2011)
    print(corpus[:10])
    model = Word2Vec(corpus, sg=1, min_count = 5, size=400, iter=30)
    print(len(list(model.wv.vocab)))
    model.wv.save_word2vec_format('word2vec.bin')
    
if __name__ == "__main__":
   main()
