import os
import pickle
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        # Load the model
        model = pickle.load(open("models/titanic_model.sav", 'rb'))

        # Process the input data and make predictions
        data = [[pclass, sex, age, sibsp, parch, fare, embarked]]
        predictions = model.predict(data)

        # Determine the prediction result
        if predictions[0] == 1:
            result = "This passenger survived! :)"
        else:
            result = "This passenger did not survived ... :("

        # Render the template with the prediction result
        return render_template('predict.html', result=result)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)

