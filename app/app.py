from flask import Flask, render_template,request
import joblib

app=Flask(__name__)

model=joblib.load('../models/fake_news_model.pkl')
vectorizer=joblib.load('../models/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['news']
    transformed_text=vectorizer.transform([text])
    prediction=model.predict(transformed_text)[0]
    result='Fake News' if prediction==0 else 'Real News'
    return render_template('index.html',prediction=result)

if __name__=='__main__':
    app.run(debug=True)