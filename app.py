# -*- coding: utf-8 -*-
from flask import Flask,request, render_template
import numpy as np
import nltk
nltk.download('punkt')
import joblib
app = Flask(__name__)
model=joblib.load("model_save.pkl")
@app.route('/')
def home():
    return render_template('index1.html')
@app.route('/predict',methods=['POST'])
def predict():
    input_value=request.form['u']
    sentences=nltk.sent_tokenize(input_value)
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    output1=model.transform(sentences)
    output2=np.array(output1.sum(axis=1)).ravel()
    N=3
    output3= [sentences[ind] for ind in np.argsort(output2, axis=0)[::-1][:N]]
    output4=[(sentence,sentence_organizer[sentence]) for sentence in output3]
    output4 = sorted(output4, key = lambda x: x[1])
    output_main = [element[0] for element in output4]
    summary = " ".join(output_main)
    return render_template('index1.html', prediction_text=summary)

if __name__ == "__main__":
    app.run()
    