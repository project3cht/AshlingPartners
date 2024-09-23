from flask import Flask, jsonify, request
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Initialize Flask app
app = Flask(__name__)

# Set the logger level for Flask's logger
app.logger.setLevel(logging.INFO)

# Initialize a simple Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Sample data for training the model (replace with your actual data)
X_train = np.random.rand(100, 4)
y_train = np.random.randint(0, 2, 100)
model.fit(X_train, y_train)

@app.route('/')
def hello():
    app.logger.info('Main endpoint processing HTTP request')
    return jsonify({"success":True, "message": "Hello, World!"})


@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Prediction endpoint processing HTTP request')
    
    # Input validation
    if not request.json or 'features' not in request.json:
        app.logger.error('Invalid input: missing features')
        return jsonify({"success": False, "error": "Missing features in input"}), 400
    
    features = request.json['features']
    
    # Ensure features is a list of numbers
    if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
        app.logger.error('Invalid input: features must be a list of numbers')
        return jsonify({"success": False, "error": "Features must be a list of numbers"}), 400
    
    # Ensure correct number of features
    if len(features) != 4:  # Assuming 4 features based on training data
        app.logger.error('Invalid input: incorrect number of features')
        return jsonify({"success": False, "error": "Incorrect number of features"}), 400
    
    try:
        # Make prediction
        prediction = model.predict([features])[0]
        app.logger.info(f'Prediction made: {prediction}')
        return jsonify({"success": True, "prediction": int(prediction)})
    except NotFittedError:
        app.logger.error('Model not fitted')
        return jsonify({"success": False, "error": "Model not fitted"}), 500
    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({"success": False, "error": "Error during prediction"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50505)