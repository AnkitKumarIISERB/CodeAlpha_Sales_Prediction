from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('best_sales_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values
    tv = float(request.form['TV'])
    radio = float(request.form['Radio'])
    newspaper = float(request.form['Newspaper'])

    # Make prediction
    prediction = model.predict(np.array([[tv, radio, newspaper]]))[0]
    output = round(prediction, 2)

    # Pass the prediction with a fade-in class
    return render_template('index.html',
                           prediction_text=f"💡 Predicted Sales: {output} units")

if __name__ == '__main__':
    app.run(debug=True)
