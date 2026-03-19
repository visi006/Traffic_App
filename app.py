from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'accident_severity_model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    model = None
    print("⚠️  Model file not found. Train and save the model first.")


# Label encoding maps (must match what was used during training)
ENCODING_MAPS = {
    'weather': {'Clear': 0, 'Cloudy': 1, 'Foggy': 2, 'Rainy': 3, 'Stormy': 4},
    'road_type': {'National Highway': 0, 'Rural Road': 1, 'State Highway': 2, 'Urban Road': 3},
    'road_condition': {'Dry': 0, 'Flooded': 1, 'Under Construction': 2, 'Wet': 3},
    'lighting': {'Dark': 0, 'Dawn': 1, 'Daylight': 2, 'Dusk': 3},
    'traffic_control': {'None': 0, 'Police': 1, 'Signals': 2, 'Signs': 3},
    'vehicle_type': {'Bus': 0, 'Car': 1, 'Cycle': 2, 'Motorcycle': 3, 'Pedestrian': 4, 'Truck': 5, 'Van': 6},
    'driver_gender': {'Female': 0, 'Male': 1},
    'license_status': {'Expired': 0, 'None': 1, 'Valid': 2},
    'alcohol': {'No': 0, 'Yes': 1},
    'location_type': {'Bridge': 0, 'Curve': 1, 'Intersection': 2, 'Straight Road': 3},
    'time_session': {'Afternoon': 0, 'Evening': 1, 'Morning': 2, 'Night': 3},
    'month': {
        'January': 0, 'February': 1, 'March': 2, 'April': 3,
        'May': 4, 'June': 5, 'July': 6, 'August': 7,
        'September': 8, 'October': 9, 'November': 10, 'December': 11
    },
    'day_of_week': {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
}

SEVERITY_INFO = {
    0: {
        'label': 'LOW',
        'color': 'green',
        'emoji': '🟢',
        'alert': 'Low Risk — Safe Driving Conditions',
        'tips': [
            'Continue following all traffic rules',
            'Maintain safe following distance',
            'Stay alert and focused on the road'
        ]
    },
    1: {
        'label': 'MEDIUM',
        'color': 'orange',
        'emoji': '🟡',
        'alert': 'Medium Risk — Drive With Caution',
        'tips': [
            'Reduce speed and drive carefully',
            'Maintain extra distance from vehicles ahead',
            'Avoid distractions — no phone usage',
            'Follow speed limits strictly'
        ]
    },
    2: {
        'label': 'HIGH',
        'color': 'red',
        'emoji': '🔴',
        'alert': 'HIGH RISK — Immediate Caution Required!',
        'tips': [
            'Immediately reduce speed',
            'Switch on hazard lights',
            'Pull over safely if conditions worsen',
            'Alert nearby vehicles',
            'Contact emergency services if needed: 112'
        ]
    }
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        data = request.get_json()

        # Build feature vector in the same order as training
        features = [
            int(data.get('year', 2023)),
            ENCODING_MAPS['month'].get(data.get('month', 'January'), 0),
            ENCODING_MAPS['day_of_week'].get(data.get('day_of_week', 'Monday'), 0),
            int(data.get('num_vehicles', 1)),
            ENCODING_MAPS['vehicle_type'].get(data.get('vehicle_type', 'Car'), 1),
            int(data.get('num_casualties', 0)),
            int(data.get('num_fatalities', 0)),
            ENCODING_MAPS['weather'].get(data.get('weather', 'Clear'), 0),
            ENCODING_MAPS['road_type'].get(data.get('road_type', 'Urban Road'), 3),
            ENCODING_MAPS['road_condition'].get(data.get('road_condition', 'Dry'), 0),
            ENCODING_MAPS['lighting'].get(data.get('lighting', 'Daylight'), 2),
            ENCODING_MAPS['traffic_control'].get(data.get('traffic_control', 'Signals'), 2),
            int(data.get('speed_limit', 60)),
            int(data.get('driver_age', 30)),
            ENCODING_MAPS['driver_gender'].get(data.get('driver_gender', 'Male'), 1),
            ENCODING_MAPS['license_status'].get(data.get('license_status', 'Valid'), 2),
            ENCODING_MAPS['alcohol'].get(data.get('alcohol', 'No'), 0),
            ENCODING_MAPS['location_type'].get(data.get('location_type', 'Straight Road'), 3),
            ENCODING_MAPS['time_session'].get(data.get('time_session', 'Morning'), 2),
        ]

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0].tolist()

        severity = SEVERITY_INFO[int(prediction)]
        confidence = round(max(probabilities) * 100, 1)

        return jsonify({
            'severity_level': int(prediction),
            'severity_label': severity['label'],
            'color': severity['color'],
            'emoji': severity['emoji'],
            'alert': severity['alert'],
            'tips': severity['tips'],
            'confidence': confidence,
            'probabilities': {
                'low': round(probabilities[0] * 100, 1),
                'medium': round(probabilities[1] * 100, 1),
                'high': round(probabilities[2] * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
