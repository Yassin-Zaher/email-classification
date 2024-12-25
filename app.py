from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    message_tfidf = vectorizer.transform([message])
    
    prediction = model.predict(message_tfidf)
    
    result = '🟥spam' if prediction == 1 else '✅not spam'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
