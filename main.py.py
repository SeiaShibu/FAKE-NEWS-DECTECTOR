from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import openai

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'ss'  # Needed for session management

# Load the trained model and TF-IDF vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Home route to render the index page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle input and make predictions
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        news = str(request.form['news'])
        print("Input news:", news)
        
        # Transform the input news using the TF-IDF vectorizer
        news_tfidf = vectorizer.transform([news])
        
        # Make a prediction using the loaded model
        prediction = model.predict(news_tfidf)
        prediction_label = 'REAL' if prediction[0] == 1 else 'FAKE'
        
        # Print the prediction result
        print("Prediction:", prediction_label)

        # Render a template with the prediction result
        return render_template('out.html', news=news, prediction=prediction_label)

    

    return render_template('out.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input news article from the form
    news = request.form.get('news', '')
    if not news:
        return redirect(url_for('out'))

    try:
        # Preprocess and predict using the ML model
        news_tfidf = vectorizer.transform([news])
        prediction = model.predict(news_tfidf)[0]
        label = "REAL" if prediction == 1 else "FAKE"

        # Use OpenAI API for additional reasoning
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI news analyst. Analyze the following news article and provide reasoning on whether it is real or fake:"},
                {"role": "user", "content": news}
            ]
        )
        reasoning = openai_response['choices'][0]['message']['content']

        # Store the result in session for later use in out.html
        session['label'] = label
        session['reasoning'] = reasoning

        # Redirect to the out.html page
        return redirect(url_for('out'))

    except Exception as e:
        # Handle any errors
        return redirect(url_for('out'))

# Route to display results on out.html
@app.route('/out')
def out():
    # Get the result from session
    label = session.get('label', 'Unknown')
    reasoning = session.get('reasoning', 'No reasoning available')

    return render_template('out.html', label=label, reasoning=reasoning)

if __name__ == '__main__':
    app.run(debug=True)
