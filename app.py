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


@app.route('/dataset')
def dataset():
    import pandas as pd

    df = pd.read_csv(os.path.join(BASE, 'Zomato Dataset.csv'))

    # Basic statistics
    stats = {
        'total_rows'     : 45584,
        'cleaned_rows'   : 42592,
        'total_features' : 20,
        'target'         : 'time_taken_min',
        'missing_before' : int(df.isnull().sum().sum()),
        'avg_delivery'   : round(df['Time_taken (min)'].mean(), 1),
        'min_delivery'   : int(df['Time_taken (min)'].min()),
        'max_delivery'   : int(df['Time_taken (min)'].max()),
    }

    # Feature list
    features = df.columns.tolist()

    # Sample rows — first 20
    sample = df.head(20).to_html(
        classes='table table-sm table-bordered table-hover',
        index=False,
        border=0
    )

    # Missing values per column
    missing = df.isnull().sum()
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ['Column', 'Missing Count']
    missing_df['Missing %'] = ((missing_df['Missing Count'] / len(df)) * 100).round(2)
    missing_table = missing_df.to_html(
        classes='table table-sm table-bordered',
        index=False,
        border=0
    )

    return render_template('dataset.html',
                           stats=stats,
                           features=features,
                           sample=sample,
                           missing_table=missing_table)


@app.route('/dashboard')
def dashboard():
    import pandas as pd
    import json

    df = pd.read_csv(os.path.join(BASE, 'Zomato Dataset.csv'))
    df.columns = df.columns.str.strip()

    # Avg delivery time by traffic density
    traffic_avg = df.groupby('Road_traffic_density')['Time_taken (min)'].mean().round(1)
    traffic_data = {
        'labels': traffic_avg.index.tolist(),
        'values': traffic_avg.values.tolist()
    }

    # Avg delivery time by weather
    weather_avg = df.groupby('Weather_conditions')['Time_taken (min)'].mean().round(1)
    weather_data = {
        'labels': weather_avg.index.tolist(),
        'values': weather_avg.values.tolist()
    }

    # Avg delivery time by city
    city_avg = df.groupby('City')['Time_taken (min)'].mean().round(1)
    city_data = {
        'labels': city_avg.index.tolist(),
        'values': city_avg.values.tolist()
    }

    # Avg delivery time by vehicle type
    vehicle_avg = df.groupby('Type_of_vehicle')['Time_taken (min)'].mean().round(1)
    vehicle_data = {
        'labels': vehicle_avg.index.tolist(),
        'values': vehicle_avg.values.tolist()
    }

    # Avg delivery time by festival
    festival_avg = df.groupby('Festival')['Time_taken (min)'].mean().round(1)
    festival_data = {
        'labels': festival_avg.index.tolist(),
        'values': festival_avg.values.tolist()
    }

    # Delivery count by traffic
    traffic_count = df['Road_traffic_density'].value_counts()
    traffic_count_data = {
        'labels': traffic_count.index.tolist(),
        'values': traffic_count.values.tolist()
    }

    # Summary stats
    stats = {
        'avg_time'     : round(df['Time_taken (min)'].mean(), 1),
        'total'        : len(df),
        'max_traffic'  : traffic_avg.idxmax(),
        'worst_weather': weather_avg.idxmax()
    }

    return render_template('dashboard.html',
        traffic_data     = json.dumps(traffic_data),
        weather_data     = json.dumps(weather_data),
        city_data        = json.dumps(city_data),
        vehicle_data     = json.dumps(vehicle_data),
        festival_data    = json.dumps(festival_data),
        traffic_count    = json.dumps(traffic_count_data),
        stats            = stats
    )


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
    
    
