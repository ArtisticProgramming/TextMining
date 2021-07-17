from flask import Flask
app = Flask(__name__)
import module as m
ml = m.Learn();




@app.route('/')
def hello():
    return "Hello World!"



@app.route('/svm/<text>')
def svm(text):
    resultSVM = ml.SVM.predict([text])
    return str(resultSVM)
    
@app.route('/nb/<text>')
def nb(text):
    resultNB = ml.NB.predict([text])
    return str(resultNB) 
    
if __name__ == '__main__':
    app.run()