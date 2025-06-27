# app.py
import pickle
from flask import Flask, request, render_template

# Load model and vectorizer
model = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['tweet']
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    label = "Hate Speech ❌" if prediction == 1 else "Non-Hate ✅"
    return render_template('index.html', prediction=label, tweet=text)

if __name__ == '__main__':
    app.run(debug=True)
