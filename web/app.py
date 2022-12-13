from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from preprocess import Preprocessing, Metric
import subprocess
from keras.models import load_model
from classification import Classifier
import numpy as np
from keras.models import Sequential, Model

app = Flask(__name__)
app.secret_key = '1_x_@ifu+*k#(t!r(1)zm89+tuof@pum30k^t+2ct!23_c6l-)'
client = MongoClient('localhost', 27017)
db = client.flaskdbs
student_col = db.student

predicted_student = {}


data_train = None
inter_layer = None
neural_network = None

def coded(name):
    length = len(name) - 3;
    return name.replace(name[3:len(name)], '*' * length)

def init():
    global data_train, inter_layer, neural_network
    merge_data = Preprocessing()
    merge_data.get_merge_data('2010', 'allpost')
    data_train = list(merge_data.dataset.items())
    custom = {'msfe':Metric.msfe, 'precision':Metric.precision, 'recall':Metric.recall, 'f1_score':Metric.f1_score}
    CNN_model = load_model('static/model/CNN.h5',custom_objects=custom)
    inter_layer = Model(inputs=CNN_model.input, outputs=CNN_model.get_layer('penultimate').output)
    neural_network = load_model('static/model/neural_model.h5', custom_objects=custom)  
    

'''@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)'''
def combine(NN_pred, RF_pred):
    NN_pred = np.array([i[0] for i in NN_pred])
    target_pred = []
    for i in range(len(RF_pred)):
        if NN_pred[i] == RF_pred[i]:
            target_pred.append(NN_pred[i])
        else:
            target_pred.append(1)
    return target_pred
    

def tokenize(student_pre):
    for student in student_pre:
        posts = ''
        for post in student['posts']:
            posts += post['title'] + ' ' + post['content']
        file = open('static/resources/posts/' + student['MSSV'] + '.txt', 'w', encoding="utf-8")
        file.write(posts)
    cmd = "vnTokenizer.bat -i ../resources/posts -o ../resources/token"
    run_batch = subprocess.Popen(cmd, shell=True, cwd='static/tokenize', stdout=subprocess.PIPE)
    run_batch.wait()

def make_prediction(student_pre):
    tokenize_post = []
    numeric = []
    predict_list = []
    for student in student_pre:
        numeric.append(student['numeric'])
        predict_list.append(student['MSSV'])
        file = open('static/resources/token/' + student['MSSV'] + '.txt', 'r', encoding="utf-8")
        post = file.read()
        tokenize_post.append(post)            
    numeric = np.array(numeric)                 
    post_vec = Preprocessing.word2vec(tokenize_post)
    Preprocessing.sparness(post_vec, numeric, data_train)
        
                   
    feature_vector = inter_layer.predict(post_vec)
    merge_vector = Preprocessing.merge_data_feature(feature_vector, numeric, 20)
    pred = neural_network.predict_classes(merge_vector)
    RandomForest = Classifier()
    RandomForest.RandomForest('static/model/Random_Forest.sav')
    RF_pred = RandomForest.model.predict(numeric)
    target_pred = combine(pred, RF_pred)
    predicted_student = dict(zip(predict_list, target_pred))
    return predicted_student

def authentical(role):
    if 'username' not in session:
        return True
    if role != session['role']:
        return True
    return False

def filt(filt):
    items = predicted_student.items()
    students = list(filter(lambda x: x[1] == filt, items))
    students = [i[0] for i in students]
    return students
        
@app.route("/students", methods = ['POST', 'GET'])
def students():
    if authentical('teacher'):
        return redirect(url_for('login'))
    CUsername = session['username']
    infos = []
    predict_list = []
    url = 'prediction.html'
    if request.method == "POST":
        global predicted_student
        if 'clear_all' in request.form:
            predicted_student = {}
        else:
            predict_list = request.form.getlist("predict")
            student_pre = list(student_col.find({"MSSV":{"$in": predict_list}}))
            tokenize(student_pre)
            new_dict = make_prediction(student_pre)
            predicted_student.update(new_dict)
                           
        #file = open('static/resources/token.txt', 'r', encoding="utf-8")
        #print(file.read())

    query = {}
    filt_student = request.args.get('filter')
    if filt_student != None:
        students = filt(int(filt_student))
        if int(filt_student) == 1:
            title = "Student List: Fail"
        else:
            title = "Student List: Pass"
        query = {"MSSV":{"$in": students}}
        url = 'read.html'
            
    for student in student_col.find(query):
        scores = student['numeric'][0:-2]
        scores.insert(0, int(student['numeric'][-1]))
        prediction = predicted_student.get(student['MSSV'], None)
        if prediction != None:
            if prediction == 0:
                predict = "Đậu"
            else:
                predict = "<span style='color:red'>Rớt</span>"
        else:
            predict = "<input class='predict2' type='checkbox' name='predict' value='" + student['MSSV'] + "'>"
        info = (student['MSSV'], '*' * len(student['username']), scores, predict, coded(student['MSSV']))
        infos.append(info)

    return render_template(url, **locals())

@app.route("/", methods = ['POST', 'GET'])
def login():
    if request.method == "POST":
        error = 'Username is wrong'
        username = request.form['user']
        password = request.form['pass']
        teacher = db.teacher.find_one({'username':username})
        if teacher != None:
            if password == teacher['password']:
                session["username"] = username
                session["role"] = "teacher"
                return redirect(url_for('students'))
            else:
                error = 'Password incorrect'
        else:
            student = student_col.find_one({'MSSV':username})
            if student != None:
                if password =='pwd':
                    session["username"] = username
                    session["role"] = "student"
                    return redirect(url_for('profile'))
                else:
                    error = 'Password incorrect'
        return render_template('login.html', error=error)
    return render_template('login.html')


@app.route("/students/<string:name>/")
def getStudent(name):
    if authentical('teacher'):
        return redirect(url_for('login'))
    CUsername = session['username']
    student = student_col.find_one({"MSSV":name})

    MSSV = coded(student['MSSV'])
    UserName = '*' * len(student['username'])
    numeric = student['numeric']
    scores = numeric[0:-2]
    NumPost = int(numeric[-1])
    posts = student['posts']
    prediction = predicted_student.get(student['MSSV'], None)
    if prediction != None:
        if prediction == 1:
            predict = "<span style='color:red'>Rớt</span>"
        else:
            predict = "Đậu"
    return render_template('student.html', **locals())

@app.route("/statistic")
def statistic():
    if authentical('teacher'):
        return redirect(url_for('login'))
    CUsername = session['username']
    values = predicted_student.values()
    total = len(values)
    fail_student = sum(values)
    pass_student = total - fail_student
    return render_template('statistic.html', **locals())


@app.route("/profile")
def profile():
    if authentical('student'):
        return redirect(url_for('login'))
    name = session['username']
    student = student_col.find_one({"MSSV":name})
    
    MSSV = coded(student['MSSV'])
    UserName = '*' * len(student['username'])
    numeric = student['numeric']
    scores = numeric[0:-2]
    NumPost = int(numeric[-1])
    posts = student['posts']
    prediction = predicted_student.get(student['MSSV'], None)
    if prediction != None:
        if prediction == 1:
            warning = '<strong>Bạn có thể rớt</strong>.<br> '
            if len(list(filter(lambda x: x <= 5, scores))) > 0:
                warning += 'Một số bài tập bạn có điểm khá thấp.<br>'
            if NumPost <=3:
                warning += 'Bạn có vẻ không tham gia diễn đàn nhiều lắm. Diễn đàn là nơi tốt để bạn có thể trao đổi các vấn đề khó khăn với sinh viên và giảng viên.<br>'
            warning += 'Bạn cố gắng lên nhé.'
    return render_template('profile.html', **locals())

@app.route("/logout")
def logout():
    session.pop("username")
    session.pop("role")
    return redirect(url_for("login"))

if __name__ == '__main__':
    init()
    app.run()
