from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


# Load the models
model = joblib.load('models/best_naive_bayes_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess and predict
    prediction = model.predict(vectorizer.transform([text]))[0]
    if prediction == 1:
        sentiment = "Positive"
    elif prediction == -1:
        sentiment = "Negative"
    elif prediction == 0:
        sentiment = "Neutral"
    return jsonify({'prediction': sentiment})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
