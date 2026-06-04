from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load model files once at startup ─────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

model            = joblib.load(os.path.join(BASE, 'model/best_model.pkl'))
preprocessor     = joblib.load(os.path.join(BASE, 'model/preprocessor.pkl'))
selected_indices = np.load(os.path.join(BASE, 'model/selected_indices.npy'))

print("✓ Model loaded successfully")

# ── Input columns (must match your X before preprocessing) ───────
INPUT_COLUMNS = [
    'delivery_person_age', 'delivery_person_ratings',
    'restaurant_latitude', 'restaurant_longitude',
    'delivery_location_latitude', 'delivery_location_longitude',
    'weather_conditions', 'road_traffic_density',
    'vehicle_condition', 'type_of_order', 'type_of_vehicle',
    'multiple_deliveries', 'festival', 'city',
    'prep_time_min', 'order_hour', 'day_of_week',
    'is_weekend', 'peak_hour', 'distance_km'
]

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    import json

    # Feature importance from your RF model used in feature selection
    feature_importance_data = {
        'features': [
            'distance_km', 'delivery_person_ratings',
            'prep_time_min', 'multiple_deliveries',
            'delivery_person_age', 'road_traffic_density_Jam',
            'road_traffic_density_High', 'weather_conditions_Fog',
            'order_hour', 'city_Metropolitian',
            'festival_Yes', 'type_of_vehicle_motorcycle',
            'is_weekend', 'peak_hour_Evening Peak',
            'road_traffic_density_Medium', 'vehicle_condition',
            'day_of_week', 'type_of_order_Snack'
        ],
        'scores': [
            0.28, 0.18, 0.12, 0.09, 0.07, 0.06,
            0.04, 0.03, 0.03, 0.02, 0.02, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.005, 0.005
        ]
    }

    return render_template('features.html',
                           data=json.dumps(feature_importance_data))

# ── Prediction API ────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build input dataframe
        input_df = pd.DataFrame([{
            'delivery_person_age'         : float(data.get('delivery_person_age', 28)),
            'delivery_person_ratings'     : float(data.get('delivery_person_ratings', 4.5)),
            'restaurant_latitude'         : float(data.get('restaurant_latitude', 26.5)),
            'restaurant_longitude'        : float(data.get('restaurant_longitude', 80.3)),
            'delivery_location_latitude'  : float(data.get('delivery_location_latitude', 26.6)),
            'delivery_location_longitude' : float(data.get('delivery_location_longitude', 80.4)),
            'weather_conditions'          : str(data.get('weather_conditions', 'Sunny')),
            'road_traffic_density'        : str(data.get('road_traffic_density', 'Medium')),
            'vehicle_condition'           : int(data.get('vehicle_condition', 1)),
            'type_of_order'               : str(data.get('type_of_order', 'Meal')),
            'type_of_vehicle'             : str(data.get('type_of_vehicle', 'motorcycle')),
            'multiple_deliveries'         : float(data.get('multiple_deliveries', 0)),
            'festival'                    : str(data.get('festival', 'No')),
            'city'                        : str(data.get('city', 'Metropolitian')),
            'prep_time_min'               : float(data.get('prep_time_min', 10)),
            'order_hour'                  : int(data.get('order_hour', 12)),
            'day_of_week'                 : int(data.get('day_of_week', 1)),
            'is_weekend'                  : int(data.get('is_weekend', 0)),
            'peak_hour'                   : str(data.get('peak_hour', 'Off-Peak')),
            'distance_km'                 : float(data.get('distance_km', 5.0))
        }])

        # Preprocess
        X_processed = preprocessor.transform(input_df)

        # Apply hybrid feature selection
        X_hybrid = X_processed[:, selected_indices]

        # Predict
        prediction = round(float(model.predict(X_hybrid)[0]), 1)

        # Classify
        if prediction <= 25:
            status, color, icon = "Normal",       "success", "✅"
        elif prediction <= 35:
            status, color, icon = "Slight Delay", "warning", "⚠️"
        else:
            status, color, icon = "High Delay",   "danger",  "🚨"

        return jsonify({
            'success'    : True,
            'prediction' : prediction,
            'status'     : status,
            'color'      : color,
            'icon'       : icon
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
    
    
