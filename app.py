from flask import Flask, render_template, request, jsonify
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

# Load FAQs
with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

questions = [faq['question'] for faq in faq_data]
answers = [faq['answer'] for faq in faq_data]

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.json['message']
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match_idx = similarities.argmax()
    score = similarities[0][best_match_idx]

    if score < 0.3:
        response = "Sorry, I didn’t understand that. Can you rephrase?"
    else:
        response = answers[best_match_idx]

    return jsonify({"reply": response})
