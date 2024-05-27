from flask import Flask, render_template, request
import pickle
import re
from nltk import WordNetLemmatizer
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vector.pkl', 'rb') as f:
    tfid = pickle.load(f)

def data_cleaning(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')).union(set(string.ascii_lowercase), {'br'})
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def prediction(comment):
    preprocessed_comment = data_cleaning(comment)
    comment_list = [preprocessed_comment]
    comment_vector = tfid.transform(comment_list)
    predicted_sentiment = model.predict(comment_vector)[0]
    return "Positive" if predicted_sentiment == 1 else "Negative"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']
        prediction_result = prediction(comment)
        return render_template('index.html', prediction=prediction_result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
