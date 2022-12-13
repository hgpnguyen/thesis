from csv import reader, writer
import sys



def load_csv(filename):
    dataset = list()
    with open(filename, 'r', encoding="utf-8") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue;
            dataset.append(row);
    return dataset;

def write_csv(filename, data):
    with open(filename, 'w', newline='', encoding="utf-8") as f:
        csv_writer = writer(f)
        csv_writer.writerows(data)
    

def tranform(name):
    name = name.split(",", 1)
    if(len(name) == 1):
        return name[0]
    name[1] = name[1][1:len(name[1])]
    if name[0]:
        name = name[1] + " " + name[0]
    else:
        name = name[1]
    return name

def score_dict(score):
    s_dict = {}
    for i in score:
        name = tranform(i[1])
        if i[2]:
            student_score = float(i[2])
        else:
            student_score = 0
        if (student_score >= 50):
            point = 1
        else:
            point = 0   
        s_dict[name] = point
    return s_dict
    
def data_filter(data, score_dict):
    new_data = []
    '''for i in data:
        name = i[0]
        if(name[0] == " "):
            i[0] = name[1:len(name)]
        score = score_dict.get(i[0])
        if score != None:
            i.insert(1, score)
            new_data.append(i)'''
    for i in data:
        name = tranform(i[1])
        name = name.replace('_','')
        name = name.replace(' ','')
        post = score_dict.get(name)
        if post == None:
            print(name)
            new_data.append(i)
        else: new_data.append(i + post)
        
    return new_data

def post_dict(data):
    s_dict = {}
    for i in data:
        name = i[0]
        name = name.replace('_','')
        name = name.replace(' ','')
        post = 0
        if int(i[2]) > 0:
            post = 1
        s_dict[name] = [int(i[2]), post, int(i[1])]
    return s_dict

def main():
    file = '2011'
    firstFile = file + '/' + file + '_diem.csv'
    secondFile = file + '/' + 'tom_tat' + file + '.csv'
    score = load_csv(firstFile)
    title = score.pop(0)
    data = load_csv(secondFile)
    data.pop(0)
    s_dict = post_dict(data)
    print(s_dict)
    data = data_filter(score, s_dict)
    title = title + ['So post', 'Post', 'Class']
    print(data[0:5])
    data.insert(0, title)
    
    outfile = file + '/post_diem' + file + '.csv'

    write_csv(outfile, data)
    


if __name__ == "__main__":
   main()


