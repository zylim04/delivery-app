from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

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

@app.route('/trends')
def trends():
    import pandas as pd
    import json

    df = pd.read_csv(os.path.join(BASE, 'Zomato Dataset.csv'))
    df.columns = df.columns.str.strip()

    # Clean target column
    df['Time_taken (min)'] = pd.to_numeric(
        df['Time_taken (min)'], errors='coerce'
    )
    df = df.dropna(subset=['Time_taken (min)'])

    # Delay classification
    def classify(t):
        if t <= 25:   return 'Normal'
        elif t <= 35: return 'Slight Delay'
        else:         return 'High Delay'

    df['delay_class'] = df['Time_taken (min)'].apply(classify)

    # 1. Delay classification distribution
    delay_counts = df['delay_class'].value_counts()
    delay_dist = {
        'labels': delay_counts.index.tolist(),
        'values': delay_counts.values.tolist()
    }

    # 2. Avg delivery time by traffic — sorted
    traffic_avg = df.groupby('Road_traffic_density')[
        'Time_taken (min)'
    ].mean().round(1).sort_values(ascending=False)
    traffic_data = {
        'labels': traffic_avg.index.tolist(),
        'values': traffic_avg.values.tolist()
    }

    # 3. Festival vs non-festival
    festival_avg = df.groupby('Festival')[
        'Time_taken (min)'
    ].mean().round(1)
    festival_data = {
        'labels': festival_avg.index.tolist(),
        'values': festival_avg.values.tolist()
    }

    # 4. Weather delay impact
    weather_avg = df.groupby('Weather_conditions')[
        'Time_taken (min)'
    ].mean().round(1).sort_values(ascending=False)
    weather_data = {
        'labels': weather_avg.index.tolist(),
        'values': weather_avg.values.tolist()
    }

    # 5. High delay rate by city
    city_delay = df.groupby('City').apply(
        lambda x: (x['delay_class'] == 'High Delay').mean() * 100
    ).round(1).sort_values(ascending=False)
    city_delay_data = {
        'labels': city_delay.index.tolist(),
        'values': city_delay.values.tolist()
    }

    # 6. Multiple deliveries impact
    multi_avg = df.groupby('multiple_deliveries')[
        'Time_taken (min)'
    ].mean().round(1).sort_values()
    multi_data = {
        'labels': [str(int(x)) for x in multi_avg.index.tolist()],
        'values': multi_avg.values.tolist()
    }

    # Key stats for insight cards
    jam_avg    = round(df[df['Road_traffic_density'] == 'Jam'][
        'Time_taken (min)'].mean(), 1)
    low_avg    = round(df[df['Road_traffic_density'] == 'Low'][
        'Time_taken (min)'].mean(), 1)
    fest_avg   = round(df[df['Festival'] == 'Yes'][
        'Time_taken (min)'].mean(), 1)
    nofest_avg = round(df[df['Festival'] == 'No'][
        'Time_taken (min)'].mean(), 1)
    high_delay_pct = round(
        (df['delay_class'] == 'High Delay').mean() * 100, 1
    )

    stats = {
        'jam_avg'        : jam_avg,
        'low_avg'        : low_avg,
        'traffic_diff'   : round(jam_avg - low_avg, 1),
        'fest_avg'       : fest_avg,
        'nofest_avg'     : nofest_avg,
        'fest_diff'      : round(fest_avg - nofest_avg, 1),
        'high_delay_pct' : high_delay_pct
    }

    return render_template('trends.html',
        delay_dist   = json.dumps(delay_dist),
        traffic_data = json.dumps(traffic_data),
        festival_data= json.dumps(festival_data),
        weather_data = json.dumps(weather_data),
        city_delay   = json.dumps(city_delay_data),
        multi_data   = json.dumps(multi_data),
        stats        = stats
    )
    
@app.route('/map')
def map_page():
    return render_template('map.html')


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        from groq import Groq

        data       = request.get_json()
        user_msg   = data.get('message', '')
        prediction = data.get('prediction', None)
        inputs     = data.get('inputs', {})

        client = Groq(api_key=os.getenv('GROQ_API_KEY'))

        context = """You are a delivery time prediction assistant.
You help users understand food delivery time predictions made by an
XGBoost machine learning model trained on 42,592 Indian food delivery
records.

Key facts about the model:
- Best model: XGBoost tuned with RandomizedSearchCV
- Test R²: 0.8393 (explains 84% of delivery time variance)
- Test RMSE: 3.7771 minutes
- Training data: Zomato delivery dataset, India

Key factors affecting delivery time (by importance):
1. distance_km — strongest predictor
2. delivery_person_ratings — higher rating = faster delivery
3. prep_time_min — restaurant preparation time
4. multiple_deliveries — more orders = longer time
5. road_traffic_density — Jam causes most delay
6. weather_conditions — Stormy/Fog cause delays
7. festival — festival periods increase delivery time

Delay classification:
- Normal: ≤ 25 minutes
- Slight Delay: 26–35 minutes
- High Delay: > 35 minutes

Keep responses concise, helpful and friendly.
Always answer in 2–4 sentences maximum."""

        if prediction and inputs:
            context += f"""

Current prediction context:
- Predicted delivery time: {prediction} minutes
- Distance: {inputs.get('distance_km', 'N/A')} km
- Traffic: {inputs.get('road_traffic_density', 'N/A')}
- Weather: {inputs.get('weather_conditions', 'N/A')}
- Festival: {inputs.get('festival', 'N/A')}
- Vehicle: {inputs.get('type_of_vehicle', 'N/A')}
- Prep time: {inputs.get('prep_time_min', 'N/A')} min"""

        response = client.chat.completions.create(
            model    = 'llama-3.3-70b-versatile',
            messages = [
                {'role': 'system', 'content': context},
                {'role': 'user',   'content': user_msg}
            ],
            max_tokens  = 200,
            temperature = 0.7
        )

        reply = response.choices[0].message.content
        return jsonify({'success': True, 'reply': reply})

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 400

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
    
    
