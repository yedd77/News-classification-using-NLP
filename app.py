import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np


from keras.models import load_model
model = load_model('model2.h5')
import json
import random
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import re
from langdetect import detect, LangDetectException

# Load chatbot data
intents = json.loads(open('data2.json').read())
words = pickle.load(open('texts2.pkl', 'rb'))
classes = pickle.load(open('labels2.pkl', 'rb'))

# Initialize text classifier
classifier = pipeline("text-classification", model="mrm8488/bert-mini-finetuned-age_news-classification")
model_classifier = classifier.model
labels = model_classifier.config.id2label

# Helper functions for chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if not results:
        return [{"intent": "unknown", "probability": "0"}]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    if tag == "unknown":
        return "Sorry, I don't understand that. Can you rephrase?"
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Helper functions for text categorization
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = text.lower().strip()
    try:
        language = detect(text)
        if language != "en":
            raise ValueError(f"Unsupported language detected.")
    except LangDetectException:
        raise ValueError("Unable to detect language. Please ensure the text is valid.")
    return text

# Flask app
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.json
    if not data or not data.get("news_text"):
        return jsonify({"error": "News text is required"}), 400

    try:
        # Preprocess the input text
        cleaned_text = preprocess_text(data["news_text"])
    except ValueError as e:
        # Handle any exceptions raised during preprocessing
        return jsonify({"error": str(e)}), 400

    # Perform classification
    result = classifier(cleaned_text, truncation=True)
    return jsonify({"category": result[0]["label"]})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
