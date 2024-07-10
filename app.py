import pandas as pd
from flask import Flask,render_template,request
app=Flask(__name__)
import pickle
import joblib
model=pickle.load(open("Naive_bayes.pkl",'rb'))
ct=joblib.load('imp_scale')

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/success',methods=['POST'])
def success():
    if request.method == 'POST':
        f=request.files['file']
        data=pd.read_csv(f)
        data.drop(['Date', 'Machine_ID'], axis = 1,inplace=True)

        data1=data.to_numpy()
        y2=pd.DataFrame(model.predict(ct.transform(data1)),columns=['Downtime'])

        data['Downtime']=y2
        data.to_csv('test_file.csv',index=False)
        return render_template("data.html",Z="Your results are here.Also refer to 'test_file.csv' file generated in the current working dictionary",Y=data.to_html())

if __name__=='__main__':
    app.run(debug=True)










