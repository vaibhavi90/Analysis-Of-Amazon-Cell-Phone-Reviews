from flask import Flask, request, render_template, redirect, url_for
import pickle
import random

# Initialize Flask application
app = Flask(__name__)

# Load the model and vectorizer
with open("C:\\Users\\VAIBHAVI\\Desktop\\aacpr\\random_forest_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("C:\\Users\\VAIBHAVI\\Desktop\\aacpr\\tfidf_vectorizer (2).pkl", 'rb') as f:
    vectorizer = pickle.load(f)

# Function to map sentiment to a numeric rating
def sentiment_to_rating(sentiment):
    if sentiment == 'negative':
        return random.randint(1, 2)  # Randomly pick 1 or 2
    elif sentiment == 'neutral':
        return 3  # Fixed rating for neutral
    else:
        return random.randint(4, 5)  # Randomly pick 4 or 5 for positive

# Route for the welcome page
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

# Redirect root URL to the welcome page
@app.route('/')
def index():
    return redirect(url_for('welcome'))

# Route for the analysis homepage
@app.route('/analyze', methods=['GET', 'POST'])
def home():
    sentiment = None
    predicted_rating = None

    if request.method == 'POST':
        # Get the input review
        review = request.form['review']

        # Preprocess the input review
        review_vector = vectorizer.transform([review])

        # Make a prediction
        predicted_sentiment = model.predict(review_vector)[0]  # This returns 'positive', 'neutral', or 'negative'

        # Convert sentiment to numeric rating
        predicted_rating = sentiment_to_rating(predicted_sentiment)

        # Capitalize the sentiment label
        sentiment = predicted_sentiment.capitalize()

    return render_template('index.html', sentiment=sentiment, predicted_rating=predicted_rating)

if __name__ == "__main__":
    app.run(debug=True)
