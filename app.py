from flask import Flask, request, jsonify, render_template, flash
import pandas as pd
import os
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
from datetime import timedelta
from trade import open_trade, close_trade, get_open_trades, close_open_trades  # Import your new function

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

load_dotenv()
api_key = os.getenv('binance_key')
api_secret = os.getenv('binance_secret')
client = Client(api_key, api_secret, testnet=True)
trade_record_file = 'data/trades/trade_records.csv'
os.makedirs(os.path.dirname(trade_record_file), exist_ok=True)
if not os.path.exists(trade_record_file):
    pd.DataFrame(columns=['Symbol', 'Side', 'Entry Price', 'Exit Price', 'PnL', 'Position', 'Time']).to_csv(trade_record_file, index=False)

@app.route('/', methods=['GET'])
def home():
    # Logic for the home page (your existing code)
    df = pd.read_csv(f'data/models/metadata/metadata.csv')

    # Convert prediction time columns to datetime
    df['Current Prediction Time'] = pd.to_datetime(df['Current Prediction Time'])
    df['Next Prediction Time'] = pd.to_datetime(df['Next Prediction Time'])

    # Adjust times based on the timeframe
    for i in range(len(df)):
        time_frame = df.at[i, 'Timeframe']
        # Convert timeframe to a timedelta object
        if time_frame == '1h':
            delta = timedelta(hours=1)
        elif time_frame == '4h':
            delta = timedelta(hours=4)
        elif time_frame == '6h':
            delta = timedelta(hours=6)
        elif time_frame == '8h':
            delta = timedelta(hours=8)
        elif time_frame == '12h':
            delta = timedelta(hours=12)
        elif time_frame == '16h':
            delta = timedelta(hours=16)
        else:
            delta = timedelta(hours=0)  # Default if no valid timeframe found

        # Subtract the timedelta from both times
        df.at[i, 'Current Prediction Time'] -= delta
        df.at[i, 'Next Prediction Time'] -= delta

    time_frames = df['Timeframe'].unique()
    time_frame_data = {}
    all_models = []  # List to hold all models for sorting later

    for time_frame in time_frames:
        frame_df = df[df['Timeframe'] == time_frame]
        signals = frame_df['Current Prediction'].to_list()

        signal_1_count = signals.count(1)
        signal_0_count = signals.count(0)
        signal_minus_1_count = signals.count(-1)
        total_signals = len(signals)
        
        signal_1_percentage = (signal_1_count / total_signals) * 100 if total_signals > 0 else 0
        signal_0_percentage = (signal_0_count / total_signals) * 100 if total_signals > 0 else 0
        signal_minus_1_percentage = (signal_minus_1_count / total_signals) * 100 if total_signals > 0 else 0

        time_frame_data[time_frame] = {
            'models': frame_df.to_dict(orient='records'),
            'signal_1_percentage': signal_1_percentage,
            'signal_0_percentage': signal_0_percentage,
            'signal_minus_1_percentage': signal_minus_1_percentage
        }

        # Collect all models for sorting
        all_models.extend(frame_df.to_dict(orient='records'))

    # Sort all models based on PNL
    sorted_models = sorted(all_models, key=lambda x: x['Total PNL'], reverse=True)

    return render_template('index.html', time_frame_data=time_frame_data, sorted_models=sorted_models)

@app.route('/details/<model>', methods=['GET'])
def details(model):
    if model+'_stats.csv' in os.listdir(f'data/models/statistics/'):
        stats = pd.read_csv(f'data/models/statistics/{model}_stats.csv').iloc[-1]
        if "Unnamed: 0" in stats:
            stats = stats.drop("Unnamed: 0", axis=0)
        stats = stats.to_dict()
        
        # Load the PnL CSV and find the date column
        pnl_df = pd.read_csv(f'data/models/ledger/{model}.csv')
        pnl = pnl_df['PNL'].dropna().to_list()
        cum_pnl = pnl_df['PNL'].cumsum().dropna().to_list()
        
        # Find the date column
        date_column = next((col for col in pnl_df.columns if col.startswith("Open time")), None)
        if date_column:
            dates = pnl_df[date_column].fillna('').to_list()  # Get the dates, filling NaN with empty strings
        else:
            dates = []  # Fallback if no date column is found

    else:
        return "Model not found", 404
    
    model_name = model
    return render_template('details.html', stats=stats, pnl=pnl, cum_pnl=cum_pnl, dates=dates, model_name=model_name)

@app.route('/trade', methods=['POST', 'GET'])
def trade():
    trades = pd.read_csv('trades.csv').to_dict(orient='records')
    #print(trades)
    if request.method == 'POST':
        trades = pd.read_csv('trades.csv').to_dict(orient='records')
        symbol = request.form["symbol"]
        side = request.form["side"]
        quantity = request.form["quantity"]
        try:
            open_trade(client, symbol, side, quantity)
            flash(f"Trade for {quantity} {symbol} successfully opened.", 'success')
        except Exception as e:
            flash(f"An error occurred while opening the trade: {e}", 'danger')
        return render_template('trade.html', trades=trades)
    else:
        return render_template('trade.html', trades=trades)

@app.route('/close_all_trades', methods=['POST'])
def close_all_trades():
    trades = pd.read_csv('trades.csv').to_dict(orient='records')
    try:
        close_open_trades(client)  # Call the function to close all trades
        flash("All open trades successfully closed.", 'success')
    except Exception as e:
        flash(f"An error occurred while closing trades: {e}", 'danger')
    
    return render_template('trade.html',trades=trades)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)