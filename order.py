from time import sleep
from binance.client import Client
import os
from dotenv import load_dotenv
import time
import csv
import pandas as pd
import math

from datetime import datetime, timedelta


load_dotenv()
api_key = os.getenv('binance_key')
api_secret = os.getenv('binance_secret')
client = Client(api_key, api_secret, testnet=True)

def open_trade(client, symbol, side, quantity, risk_percent=3, reward_percent=5, order_type='MARKET'):
    """
    Opens a trade on Binance Futures Testnet with dynamic TP and SL based on 
    risk and reward percentages.
    
    :param client: Binance Client instance
    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    :param side: 'BUY' for long, 'SELL' for short
    :param quantity: Amount of the asset to buy/sell
    :param risk_percent: Percentage of risk for stop loss
    :param reward_percent: Percentage of profit for take profit
    :param order_type: Order type, default is 'MARKET'
    :return: Response from the order, Entry price
    """
    try:
        # Fetch the latest price for the symbol
        latest_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        print(f"Latest Price: {latest_price}")

        # Calculate TP and SL based on the entry price
        if side == 'BUY':
            stop_loss = latest_price * (1 - risk_percent / 100)
            take_profit = latest_price * (1 + reward_percent / 100)
        elif side == 'SELL':
            stop_loss = latest_price * (1 + risk_percent / 100)
            take_profit = latest_price * (1 - reward_percent / 100)

        # Round quantity and prices to appropriate precision
        precision = client.futures_exchange_info()['symbols']
        for symbol_info in precision:
            if symbol_info['symbol'] == symbol:
                quantity_precision = symbol_info['filters'][2]['stepSize']
                stop_loss_precision = symbol_info['filters'][0]['tickSize']
                take_profit_precision = symbol_info['filters'][0]['tickSize']
                break
        
        # Round the quantity and prices
        quantity = round(quantity, int(-1 * round(math.log10(float(quantity_precision)))))
        stop_loss = round(stop_loss, int(-1 * round(math.log10(float(stop_loss_precision)))))
        take_profit = round(take_profit, int(-1 * round(math.log10(float(take_profit_precision)))))

        # Open the main trade
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity
        )
        print(f"Trade opened: {order}")

        # Set Stop Loss and Take Profit
        sl_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',  # Reverse side for SL
            type='STOP_MARKET',
            stopPrice=stop_loss,
            quantity=quantity
        )
        print(f"Stop Loss set: {sl_order}")

        tp_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',  # Reverse side for TP
            type='TAKE_PROFIT_MARKET',
            stopPrice=take_profit,
            quantity=quantity
        )
        print(f"Take Profit set: {tp_order}")

        return order, latest_price  # Return the order and the entry price
    except Exception as e:
        print(f"An error occurred while opening the trade: {e}")
        return None, None

def close_trade(client, symbol, side, quantity, order_type='MARKET'):
    """
    Closes a trade on Binance Futures Testnet.
    
    :param client: Binance Client instance
    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    :param side: 'BUY' or 'SELL' to close the trade
    :param quantity: Amount of the asset to buy/sell
    :param order_type: Order type, default is 'MARKET'
    :return: Response from the order, Exit price
    """
    try:
        # Reverse the side to close the trade
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        
        order = client.futures_create_order(
            symbol=symbol,
            side=opposite_side,
            type=order_type,
            quantity=quantity
        )
        print(f"Trade closed: {order}")

        # Fetch the latest price for the symbol as the exit price
        exit_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        print(f"Exit Price: {exit_price}")
        
        return order, exit_price  # Return the order and the exit price
    except Exception as e:
        print(f"An error occurred while closing the trade: {e}")
        return None, None

def get_open_trades(client):
    """
    Retrieves all open trades on Binance Futures Testnet.
    
    :param client: Binance Client instance
    :return: List of open trades
    """
    try:
        positions = client.futures_position_information()
        open_trades = [position for position in positions if float(position['positionAmt']) != 0]
        
        if open_trades:
            print("Open Trades:")
            for trade in open_trades:
                print(f"Symbol: {trade['symbol']}, Position Amount: {trade['positionAmt']}, Entry Price: {trade['entryPrice']}")
        else:
            print("No open trades.")
        
        return open_trades
    except Exception as e:
        print(f"An error occurred while fetching open trades: {e}")
        return []


def save_trade_signal(signal):
    """Save the trade signal to a CSV file."""
    try:
        with open('trade_signals.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([signal])
        print(f"Signal '{signal}' saved to trade_signals.csv")
    except Exception as e:
        print(f"Error saving signal '{signal}': {e}")

def load_trade_signals():
    """Load trade signals from the CSV file."""
    if os.path.exists('trade_signals.csv'):
        with open('trade_signals.csv', mode='r') as file:
            reader = csv.reader(file)
            return [row[0] for row in reader]
    return []

def time_until_next_check():
    """Calculate the time in seconds until 11 minutes past the next hour."""
    now = datetime.now()
    next_check = now.replace(minute=11, second=0, microsecond=0)
    
    if now.minute >= 11:
        next_check += timedelta(hours=1)
    
    sleep_time = (next_check - now).total_seconds()
    return sleep_time

def fetch_trade_history(client, symbol):
    """
    Fetch all trade history for a given symbol from Binance Futures Testnet
    and save it to a CSV file.
    
    :param client: Binance Client instance
    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    """
    try:
        # Fetch trade history
        trades = client.futures_account_trades(symbol=symbol)
        
        # Convert to DataFrame
        df_trades = pd.DataFrame(trades)
        
        # Save to CSV
        csv_file = 'trade_history.csv'
        if os.path.exists(csv_file):
            df_trades.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_trades.to_csv(csv_file, index=False)
        
        print(f"Trade history saved to {csv_file}")
    except Exception as e:
        print(f"An error occurred while fetching trade history: {e}")


        
def save_trade_details(entry_price, exit_price, pnl, direction):
    """Save the trade details (entry price, exit price, PnL, direction) to a CSV file."""
    try:
        file_exists = os.path.isfile('trade_details.csv')
        with open('trade_details.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the header only if the file is new
            if not file_exists:
                writer.writerow(['Entry Price', 'Exit Price', 'PnL', 'Direction'])
            writer.writerow([entry_price, exit_price, pnl, direction])
        print(f"Trade details saved to trade_details.csv")
    except Exception as e:
        print(f"Error saving trade details: {e}")

def trade_bot(client):
    """
    Function to run the trading bot that checks for signals, opens trades, 
    and manages open positions on Binance Futures Testnet.
    """
    trade_open_signals = load_trade_signals()
    entry_price = None  # Initialize entry_price to None

    while True:
        # Get the latest signals from the API
        signals = pd.read_csv('data/models/current status/Aravos_status.csv')
        side = signals['Current Prediction'].iloc[-1]
        open_trades = get_open_trades(client)

        if len(open_trades) == 0:
            if side == 1:
                order, entry_price = open_trade(client, symbol='BTCUSDT', side='BUY', quantity=0.004, 
                                                risk_percent=3, reward_percent=5)
                if order and entry_price:
                    trade_open_signals.append('BUY')
                    save_trade_signal('BUY')
            elif side == -1:
                order, entry_price = open_trade(client, symbol='BTCUSDT', side='SELL', quantity=0.004, 
                                                risk_percent=3, reward_percent=5)
                if order and entry_price:
                    trade_open_signals.append('SELL')
                    save_trade_signal('SELL')
        else:
            last_trade = trade_open_signals[-1]
            if last_trade == 'BUY' and side == -1:
                close_order, exit_price = close_trade(client, symbol='BTCUSDT', side='SELL', quantity=0.004)
                pnl = (exit_price - entry_price) * 0.004  # Simple PnL calculation
                save_trade_details(entry_price, exit_price, pnl, 'BUY')  # Save details after closing the trade
                entry_price = None  # Reset entry price after closing the trade
                order, entry_price = open_trade(client, symbol='BTCUSDT', side='SELL', quantity=0.004, 
                                                risk_percent=3, reward_percent=5)
                if order and entry_price:
                    trade_open_signals.append('SELL')
                    save_trade_signal('SELL')
            elif last_trade == 'SELL' and side == 1:
                close_order, exit_price = close_trade(client, symbol='BTCUSDT', side='BUY', quantity=0.004)
                pnl = (entry_price - exit_price) * 0.004  # Simple PnL calculation
                save_trade_details(entry_price, exit_price, pnl, 'SELL')  # Save details after closing the trade
                entry_price = None  # Reset entry price after closing the trade
                order, entry_price = open_trade(client, symbol='BTCUSDT', side='BUY', quantity=0.004, 
                                                risk_percent=3, reward_percent=5)
                if order and entry_price:
                    trade_open_signals.append('BUY')
                    save_trade_signal('BUY')

        # Calculate time until 11 minutes past the next hour
        sleep_duration = time_until_next_check()
        print(f"Sleeping for {sleep_duration / 60:.2f} minutes until next check.")
        print('Current time:', time.ctime())
        time.sleep(sleep_duration)

if __name__ == "__main__":
    trade_bot(client)
