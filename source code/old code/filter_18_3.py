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

def write_csv(filename, data, item):
    with open(filename, 'w', newline='', encoding="utf-8") as f:
        csv_writer = writer(f)
        csv_writer.writerows(data)
        csv_writer.writerows(item)
    

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
    for i in data:
        name = i[0]
        if(name[0] == " "):
            i[0] = name[1:len(name)]
        score = score_dict.get(i[0])
        if score != None:
            i.insert(1, score)
            new_data.append(i)
    return new_data


def main():
    score = load_csv('2010/2010_diem.csv')
    score.pop(0)
    s_dict = score_dict(score)
    item = [[x[0], x[1], "", "", ""] for x in s_dict.items()]
    print(item)
    data = load_csv('2010/data2010.csv')
    #data.pop(0)
    data = data_filter(data, s_dict)
    print(len(data))

    file = "2010/demo2010_2.csv"

    write_csv(file, data, item)
    


if __name__ == "__main__":
   main()


