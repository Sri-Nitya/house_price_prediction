from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import logging

# Load the trained model
model = pickle.load(open('xgb_model.pkl', 'rb'))
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

@app.route('/')
def home():
    return render_template_string(open('templates/index.html').read())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        if request.form:
            features = {
                'MedInc': float(request.form['MedInc']),
                'HouseAge': float(request.form['HouseAge']),
                'AveRooms': float(request.form['AveRooms']),
                'Latitude': float(request.form['Latitude']),
                'Longitude': float(request.form['Longitude'])
            }
        else:
            features = request.get_json()
        
        # Validate input
        required_keys = ['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude']
        if not all(key in features for key in required_keys):
            missing_keys = [key for key in required_keys if key not in features]
            # Log the error for missing fields
            logging.error(f"Missing fields: {missing_keys} for input: {features}")
            return jsonify({'code': 400, 'message': f'Missing fields in input: {missing_keys}'}), 400

        # Validate latitude and longitude
        latitude = features['Latitude']
        longitude = features['Longitude']
        if not (-90 <= latitude <= 90):
            # Log the error for invalid latitude
            logging.error(f"Invalid latitude: {latitude} for input: {features}")
            return jsonify({'code': 400, 'message': f'Invalid latitude: {latitude}. It should be between -90 and 90.'}), 400
        if not (-180 <= longitude <= 180):
            # Log the error for invalid longitude
            logging.error(f"Invalid longitude: {longitude} for input: {features}")
            return jsonify({'code': 400, 'message': f'Invalid longitude: {longitude}. It should be between -180 and 180.'}), 400

        # Predict price
        input_features = np.array([features[key] for key in required_keys]).reshape(1, -1)
        predicted_price = float(model.predict(input_features)[0])
        # Log the prediction
        logging.info(f"Predicted price: {predicted_price} for input: {features}")
        return jsonify({'code': 200, 'predicted_price': predicted_price})

    except Exception as e:
        # Log any other errors
        logging.error(f"Error during prediction: {e} for input: {locals().get('features', 'Unavailable')}")
        return jsonify({'code': 500, 'message': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
