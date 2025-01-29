from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import pandas as pd
import gzip
from statsmodels.tsa.statespace.sarimax import SARIMAX
import traceback

# Load the model
with gzip.open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with gzip.open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create the Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_load():
    def get_timestamp(field_name):
        try:
            timestamp = request.form.get(field_name)
            print(f"{field_name.capitalize()}: {timestamp}")
            return timestamp
        except ValueError as e:
            return f"<h1 style='color:red'>Error parsing {field_name}: {e}</h1>"

    def get_temperature():
        try:
            temp = float(request.form.get('temp'))
            print("Temperature:", temp)
            return temp
        except (ValueError, TypeError):
            return f"<h1 style='color:red'>Invalid temperature value provided</h1>"

    def forecast_load(start_timestamp, end_timestamp, temp):
        try:
            num_predictions = pd.date_range(start=start_timestamp, end=end_timestamp, freq='h').shape[0]
            print(num_predictions)
            exogenous_data = np.array([temp] * num_predictions).reshape(-1, 1)
            print(exogenous_data)

            # Load the energy data
            endog_data = pd.read_csv("energy.csv")
            endog_data['timestamp'] = pd.to_datetime(endog_data['timestamp'])
            endog_data = endog_data.set_index('timestamp')

            # Explicitly set the frequency to hourly
            endog_data = endog_data.asfreq('h')

            # Filter training data
            train_data = endog_data[(endog_data.index >= '2012-01-01 00:00:00') & (endog_data.index <= '2014-12-31 23:00:00')]['load']

            # Fit the SARIMAX model
            order = (4, 1, 0)
            seasonal_order = (1, 1, 0, 24)
            model = SARIMAX(endog=train_data, order=order, seasonal_order=seasonal_order)
            results = model.fit()

            # Perform forecasting
            predictions = results.predict(start=start_timestamp, end=end_timestamp, exog=exogenous_data)
            print("Predictions (scaled):", predictions.values)

            # Inverse transform the predictions
            predictions_scaled = scaler.inverse_transform(predictions.values.reshape(-1, 1))
            print("Predictions (unscaled):", predictions_scaled)

            # Create a DataFrame with timestamps and loads
            timestamp_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq='h')
            results_df = pd.DataFrame(predictions.values, index=timestamp_index, columns=['load'])
            results_df.index.name = 'timestamp'

            # Format the DataFrame for display
            formatted_results = results_df.reset_index().to_string(index=False, header=True, float_format="{:.2f}".format)
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Energy Consumption</title>
                <style>
                    @keyframes borderAnimation {{
                        0% {{
                            border-color: #4CAF50;
                            box-shadow: 0 0 25px #4CAF50;
                        }}
                        25% {{
                            border-color: #FF5733;
                            box-shadow: 0  0 25px #FF5733;
                        }}
                        50% {{
                            border-color: #FFC300;
                            box-shadow: 0 0 25px #FFC300;
                        }}
                        75% {{
                            border-color: #3498DB;
                            box-shadow: 0 0 25px #3498DB;
                        }}
                        100% {{
                            border-color: #4CAF50;
                            box-shadow: 0 0 25px #4CAF50;
                        }}
                    }}
                    @keyframes borderAnimation2 {{
                        0% {{
                            color: #FF5733;
                        }}
                        25%{{
                            color:orangered
                        }}
                        50%{{
                            color:orange;
                        }}
                        75% {{
                            color:rgb(255, 174, 0);
                        }}
                        100%{{
                            color:orangered;
                        }}
                    }}
                    body{{background-color: #1a1a1a;font-family: 'Montserrat', sans-serif;z-index:1;}}
                    h2, h1 {{
                        text-align: center;
                        width: 100%;
                        height:25px;
                        letter-spacing: 5px;
                        animation:borderAnimation2 5s infinite;
                        margin-top: 0;
                        padding-bottom: 10px;
                        font-family: 'Montserrat', sans-serif;
                        z-index:3;
                    }}
                    table {{
                        margin: 50px auto;
                        margin-top:10px;
                        border-radius: 10px;
                        background-color: #f9f9f9;
                        animation: borderAnimation 5s infinite;
                        font-family: 'Montserrat', sans-serif;
                        width: 50%;
                        z-index:2;
                        text-align: center;
                        background-color: 
                    }}
                    th, td {{
                        padding: 15px;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #4CAF50;
                        color: white;
                        border-collapse:collapse;
                        border:none;
                    }}
                    tr:hover {{background-color: #f5f5f5;}}
                </style>
            </head>
            <body>
                <table>
                    <h1>Load Prediction:</h1>
                    <tr><th style="border-top-left-radius:10px">Timestamp</th><th style="border-top-right-radius:10px">Load</th></tr>
                    {''.join(f"<tr><td>{timestamp}</td><td>{load}</td></tr>" for timestamp, load in zip(results_df.index, results_df['load']))}
                </table>
            </body>
            </html>
            """
            return html_content
        except Exception as e:
            # Enhanced error message
            return f"<h1 style='color:red'>Error during prediction: {e}</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>"

    start_timestamp = get_timestamp('s_timestamp')
    end_timestamp = get_timestamp('e_timestamp')
    temp = get_temperature()

    if isinstance(start_timestamp, str) and start_timestamp.startswith('<h1'):
        return start_timestamp
    if isinstance(end_timestamp, str) and end_timestamp.startswith('<h1'):
        return end_timestamp
    if isinstance(temp, str) and temp.startswith('<h1'):
        return temp

    return forecast_load(start_timestamp, end_timestamp, temp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, port=5001)
