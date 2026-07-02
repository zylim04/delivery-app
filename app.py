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



@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/insights')
def insights():
    import json
    import pandas as pd

    # Read the dataset once
    df = pd.read_csv(os.path.join(BASE, 'Zomato Dataset.csv'))

    # ===== DATASET tab (on raw df, BEFORE any mutation) =====
    dataset_stats = {
        'total_rows'     : 45584,
        'cleaned_rows'   : 42592,
        'total_features' : 20,
        'target'         : 'time_taken_min',
        'missing_before' : int(df.isnull().sum().sum()),
        'avg_delivery'   : round(df['Time_taken (min)'].mean(), 1),
        'min_delivery'   : int(df['Time_taken (min)'].min()),
        'max_delivery'   : int(df['Time_taken (min)'].max()),
    }
    features = df.columns.tolist()
    sample = df.head(20).to_html(
        classes='table table-sm table-bordered table-hover', index=False, border=0)
    missing = df.isnull().sum()
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ['Column', 'Missing Count']
    missing_df['Missing %'] = ((missing_df['Missing Count'] / len(df)) * 100).round(2)
    missing_table = missing_df.to_html(
        classes='table table-sm table-bordered', index=False, border=0)

    # ===== DASHBOARD tab =====
    df.columns = df.columns.str.strip()
    for col in ['Road_traffic_density', 'Weather_conditions', 'City',
                'Festival', 'Type_of_order', 'Type_of_vehicle']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=['Time_taken (min)'])

    avg_time = round(df['Time_taken (min)'].mean(), 1)
    total = len(df)

    traffic_order = ['Low', 'Medium', 'High', 'Jam']
    traffic_avg = (df.groupby('Road_traffic_density')['Time_taken (min)']
                     .mean().round(1).reindex(traffic_order))
    traffic_data = {'labels': traffic_avg.index.tolist(), 'values': traffic_avg.values.tolist()}
    jam_avg = traffic_avg.get('Jam', 0)
    low_avg = traffic_avg.get('Low', 0)
    traffic_diff = round(jam_avg - low_avg, 1)
    max_traffic = traffic_avg.idxmax()

    weather_avg = (df.groupby('Weather_conditions')['Time_taken (min)']
                     .mean().round(1).sort_values(ascending=False))
    weather_data = {'labels': weather_avg.index.tolist(), 'values': weather_avg.values.tolist()}
    worst_weather = weather_avg.idxmax()

    city_avg = df.groupby('City')['Time_taken (min)'].mean().round(1)
    city_data = {'labels': city_avg.index.tolist(), 'values': city_avg.values.tolist()}

    festival_avg = df.groupby('Festival')['Time_taken (min)'].mean().round(1)
    nofest_avg = festival_avg.get('No', 0)
    fest_avg = festival_avg.get('Yes', 0)
    fest_diff = round(fest_avg - nofest_avg, 1)
    festival_data = {'labels': ['Non-Festival', 'Festival'], 'values': [nofest_avg, fest_avg]}

    df['order_hour'] = pd.to_datetime(df['Time_Orderd'], errors='coerce').dt.hour
    hour_avg = (df.groupby('order_hour')['Time_taken (min)'].mean().round(1).sort_index())
    hour_data = {'labels': [f"{int(h):02d}:00" for h in hour_avg.index.tolist()],
                 'values': hour_avg.values.tolist()}

    df['order_date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
    df['day_of_week'] = df['order_date'].dt.day_name()
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_avg = (df.groupby('day_of_week')['Time_taken (min)'].mean().round(1).reindex(day_order))
    day_data = {'labels': day_avg.index.tolist(), 'values': day_avg.fillna(0).values.tolist()}

    multi_avg = (df.groupby('multiple_deliveries')['Time_taken (min)'].mean().round(1).sort_index())
    multi_data = {'labels': [str(int(m)) for m in multi_avg.index.tolist()],
                  'values': multi_avg.values.tolist()}

    def classify_delay(t):
        if t <= 25: return 'Normal'
        elif t <= 35: return 'Slight Delay'
        else: return 'High Delay'

    df['delay_class'] = df['Time_taken (min)'].apply(classify_delay)
    delay_order = ['Normal', 'Slight Delay', 'High Delay']
    delay_counts = (df['delay_class'].value_counts().reindex(delay_order).fillna(0))
    delay_dist = {'labels': delay_order, 'values': delay_counts.values.tolist()}
    high_delay_pct = round((df['delay_class'] == 'High Delay').mean() * 100, 1)

    dashboard_stats = {
        'avg_time': avg_time, 'total': f"{total:,}", 'max_traffic': max_traffic,
        'worst_weather': worst_weather, 'high_delay_pct': high_delay_pct,
        'jam_avg': jam_avg, 'low_avg': low_avg, 'traffic_diff': traffic_diff,
        'fest_avg': fest_avg, 'nofest_avg': nofest_avg, 'fest_diff': fest_diff
    }

    # ===== FEATURES tab =====
    shap_path = os.path.join(BASE, 'model/shap_values.json')
    rf_path   = os.path.join(BASE, 'model/rf_importance.json')
    if os.path.exists(shap_path):
        with open(shap_path) as f: shap_data = json.load(f)
        use_shap = True
    else:
        shap_data = {'features': [], 'values': []}; use_shap = False
    if os.path.exists(rf_path):
        with open(rf_path) as f: rf_data = json.load(f)
        use_rf = True
    else:
        rf_data = {'features': [], 'values': []}; use_rf = False

    return render_template('insights.html',
        dataset_stats=dataset_stats, features=features,
        sample=sample, missing_table=missing_table,
        dashboard_stats=dashboard_stats,
        traffic_data=json.dumps(traffic_data),
        weather_data=json.dumps(weather_data),
        city_data=json.dumps(city_data),
        festival_data=json.dumps(festival_data),
        hour_data=json.dumps(hour_data),
        day_data=json.dumps(day_data),
        multi_data=json.dumps(multi_data),
        delay_dist=json.dumps(delay_dist),
        shap_data=json.dumps(shap_data),
        rf_data=json.dumps(rf_data),
        use_shap=use_shap, use_rf=use_rf)


# Keep old URLs working — redirect to the merged Insights tabs
@app.route('/dashboard')
def dashboard():
    from flask import redirect, url_for
    return redirect(url_for('insights') + '#dashboard')

@app.route('/dataset')
def dataset():
    from flask import redirect, url_for
    return redirect(url_for('insights') + '#dataset')

@app.route('/performance')
def performance():
    from flask import redirect, url_for
    return redirect(url_for('insights') + '#performance')

@app.route('/features')
def features():
    from flask import redirect, url_for
    return redirect(url_for('insights') + '#features')





# Redirect old /trends links to the merged dashboard
@app.route('/trends')
def trends():
    from flask import redirect, url_for
    return redirect(url_for('dashboard'))


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
LightGBM machine learning model trained on 42,592 Indian food delivery
records.

Key facts about the model:
- Best model: LightGBM tuned with Optuna (Bayesian optimization)
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
    

@app.route('/admin')
def admin():
    import json
    model_info = {
        'name'      : 'LightGBM (Tuned - Optuna)',
        'test_r2'   : 0.8393,
        'test_rmse' : 3.7771,
        'test_mae'  : 3.0418,
        'trained_on': '42,592 records',
        'features'  : 17,
        'file'      : 'best_model.pkl'
    }
    return render_template('admin.html', model_info=model_info)

@app.route('/download-model')
def download_model():
    from flask import send_file
    return send_file(
        os.path.join(BASE, 'model/best_model.pkl'),
        as_attachment=True,
        download_name='best_model_xgboost.pkl'
    )
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    import csv
    from datetime import datetime
    data     = request.get_json()
    name     = data.get('name', '')
    email    = data.get('email', '')
    message  = data.get('message', '')
    rating   = data.get('rating', '')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    feedback_file = os.path.join(BASE, 'feedback.csv')
    file_exists   = os.path.exists(feedback_file)

    with open(feedback_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp','Name','Email','Rating','Message'])
        writer.writerow([timestamp, name, email, rating, message])

    return jsonify({'success': True, 'message': 'Thank you for your feedback!'})


@app.route('/api/prediction-context', methods=['POST'])
def prediction_context():
    try:
        import pandas as pd
        data = request.get_json()

        df = pd.read_csv(os.path.join(BASE, 'Zomato Dataset.csv'))
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Time_taken (min)'])

        traffic  = data.get('road_traffic_density', '')
        weather  = data.get('weather_conditions', '')
        city     = data.get('city', '')
        festival = data.get('festival', '')

        overall_avg = round(df['Time_taken (min)'].mean(), 1)

        traffic_avg = round(
            df[df['Road_traffic_density'] == traffic]['Time_taken (min)'].mean(), 1
        )
        weather_avg = round(
            df[df['Weather_conditions'] == weather]['Time_taken (min)'].mean(), 1
        )
        city_avg = round(
            df[df['City'] == city]['Time_taken (min)'].mean(), 1
        )
        festival_avg = round(
            df[df['Festival'] == festival]['Time_taken (min)'].mean(), 1
        )

        return jsonify({
            'success': True,
            'context': {
                'labels': [
                    'Overall Avg',
                    f'{traffic} Traffic',
                    f'{weather} Weather',
                    city,
                    'Festival' if festival == 'Yes' else 'Non-Festival'
                ],
                'values': [
                    overall_avg, traffic_avg, weather_avg,
                    city_avg, festival_avg
                ]
            }
        })

    except Exception as e:
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

      
@app.route('/api/similar-cases', methods=['POST'])
def similar_cases():
    try:
        import pandas as pd
        data = request.get_json()

        df = pd.read_csv(os.path.join(BASE, 'Zomato Dataset.csv'))
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Time_taken (min)'])

        # Filter by similar conditions
        traffic = data.get('road_traffic_density', '')
        weather = data.get('weather_conditions', '')
        city    = data.get('city', '')
        dist    = float(data.get('distance_km', 5))

        # Match same traffic + city, similar distance (±3 km)
        filtered = df[
            (df['Road_traffic_density'] == traffic) &
            (df['City'] == city)
        ].copy()

        # If too few results, relax filter
        if len(filtered) < 5:
            filtered = df[
                df['Road_traffic_density'] == traffic
            ].copy()

        # Calculate distance from Haversine
        def haversine(lat1, lon1, lat2, lon2):
            import math
            R = 6371
            dLat = math.radians(lat2 - lat1)
            dLon = math.radians(lon2 - lon1)
            a = (math.sin(dLat/2)**2 +
                 math.cos(math.radians(lat1)) *
                 math.cos(math.radians(lat2)) *
                 math.sin(dLon/2)**2)
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        filtered['distance_km'] = filtered.apply(
            lambda r: haversine(
                r['Restaurant_latitude'],
                r['Restaurant_longitude'],
                r['Delivery_location_latitude'],
                r['Delivery_location_longitude']
            ) if all(pd.notna([
                r['Restaurant_latitude'],
                r['Restaurant_longitude'],
                r['Delivery_location_latitude'],
                r['Delivery_location_longitude']
            ])) else 0, axis=1
        )

        # Filter similar distance ±4 km
        similar = filtered[
            (filtered['distance_km'] >= dist - 4) &
            (filtered['distance_km'] <= dist + 4)
        ].copy()

        if len(similar) < 5:
            similar = filtered.copy()

        # Sample top 8 records
        similar = similar.sample(
            min(8, len(similar)), random_state=42
        )

        # Build result
        results = []
        for _, row in similar.iterrows():
            results.append({
                'traffic'  : str(row.get('Road_traffic_density', '')),
                'weather'  : str(row.get('Weather_conditions', '')),
                'city'     : str(row.get('City', '')),
                'vehicle'  : str(row.get('Type_of_vehicle', '')),
                'festival' : str(row.get('Festival', '')),
                'distance' : round(float(row.get('distance_km', 0)), 1),
                'actual'   : int(row.get('Time_taken (min)', 0))
            })

        return jsonify({'success': True, 'cases': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)