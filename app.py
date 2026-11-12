import json
import os
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

with open(os.path.join(BASE, "faq.json"), encoding="utf-8") as f:
    data = json.load(f)

questions = [d["question"] for d in data]
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = vectorizer.fit_transform(questions)

THRESHOLD = 0.35

def get_reply(msg):
    sims = cosine_similarity(vectorizer.transform([msg.lower()]), X)[0]
    order = sims.argsort()[::-1]
    best = order[0]
    if sims[best] >= THRESHOLD:
        return {"reply": data[best]["answer"], "suggestions": []}
    # match hoyni, top 3 close question suggest koro
    sugg = [questions[i] for i in order[:3] if sims[i] > 0.1]
    return {
        "reply": "Sorry, thik bujhte parlam na." + (" Tumi ki eta jiggesh korechile?" if sugg else ""),
        "suggestions": sugg,
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.get_json() or {}).get("message", "")
    return jsonify(get_reply(msg))

if __name__ == "__main__":
    app.run(debug=True)