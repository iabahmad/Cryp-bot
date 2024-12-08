################################################################ Libraries
# For Manipulating Date
import datetime
import time

# Library To Parse The Config File
import configparser

# Other Necessary Libraries
import pandas as pd
import numpy as np
import warnings
import joblib
import copy
import os

# Importing Custom Modules
from statistics_calculation import statistics
from technical_indicators import indicators
from utils import utils

# Ignoring all warnings from the output log
warnings.filterwarnings('ignore')


################################################################## Declarations
# Settig Up The Directory From The Directory Hierarchy
# Get the script's current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the parent directory
parent_dir = os.path.dirname(script_dir)

# define path to all the needed directories within 'parent' and 'data' directory
config_dir         = os.path.join(script_dir, "configuration")
data_dir           = os.path.join(parent_dir, "data")
ohlc_dir           = os.path.join(data_dir,   "ohlc")
models_dir         = os.path.join(data_dir,   "models")
models_info_dir    = os.path.join(models_dir, "info")
ledger_dir         = os.path.join(models_dir, 'ledger')
current_status_dir = os.path.join(models_dir, 'current status')
statistics_dir     = os.path.join(models_dir, 'statistics')
pickle_files_dir   = os.path.join(models_dir, 'pickle files')
metadata_dir       = os.path.join(models_dir, 'metadata')

# Create directories if they do not exist
os.makedirs(config_dir,         exist_ok=True)
os.makedirs(data_dir,           exist_ok=True)
os.makedirs(ohlc_dir,           exist_ok=True)
os.makedirs(models_dir,         exist_ok=True)
os.makedirs(models_info_dir,    exist_ok=True)
os.makedirs(ledger_dir,         exist_ok=True)
os.makedirs(current_status_dir, exist_ok=True)
os.makedirs(statistics_dir,     exist_ok=True)
os.makedirs(metadata_dir,       exist_ok=True)
os.makedirs(pickle_files_dir,   exist_ok=True)

# Create the full path to the metadata file within the 'metadata' directory in the 'data' directory
metadata_file = os.path.join(metadata_dir, 'metadata.csv')

# Create the full path to the config file within the 'configuration' directory relative to this path
config_file = os.path.join(config_dir, "config.ini")

# Read configuration settings
config = configparser.ConfigParser()
config.read(config_file)

# File names for data to be used and all the models to be used as well
csv_filename_1 = config['SIMULATION']['ohlc_data']
csv_filename_2 = config['SIMULATION']['model_info']

# Create the full path to the CSV file 1 and CSV file 2 in the 'ohlc' and 'info' directory within 'data' directory
full_csv_path_1 = os.path.join(ohlc_dir,        csv_filename_1)
full_csv_path_2 = os.path.join(models_info_dir, csv_filename_2)


################################################################## Functions
# Function that generates backtest dataframe
def generate_backtest_df(data, data_1m, df_for_freq_inferring, initial_balance=1000, transaction_fee=0.01):
    """
    Generate a backtesting DataFrame based on MACD signals.

    Args:
        data (pd.DataFrame): DataFrame containing historical data with columns:
                             'Open time (4H)', 'Open', 'High', 'Low', 'Close', 'Volume', 'MACD_Signal'.
                             The DataFrame must have 'Open time (4H)' as a DateTime index.
        data_1m (pd.DataFrame): DataFrame containing 1-minute interval data with 'Open' prices.
        initial_balance (float): Initial balance for the backtest.
        transaction_fee (float): Transaction fee as a percentage of the current balance for each trade.

    Returns:
        pd.DataFrame: A DataFrame containing the backtesting results with columns:
                      'Open time (4H)', 'direction', 'entry price', 'close price', 'PNL', 'Balance'.
    """
    
    def find_first_change(signal):
        # Convert the list to a numpy array
        signal_array = np.array(signal)

        # Find indices of non-zero elements
        non_zero_indices = np.flatnonzero(signal_array != 0)

        # Find where the value changes
        changes = np.where(np.diff(signal_array[non_zero_indices]))[0] + 1
        changes = np.insert(changes, 0, 0)
        
        if len(non_zero_indices) < 1:
            return signal.index[0], signal.index[-1]

        if len(changes) == 1:
            # Get the indices of the first change
            first_change_start = non_zero_indices[changes[0]]
            first_change_end = None
        else:
            # Get the indices of the first change
            first_change_start = non_zero_indices[changes[0]]
            first_change_end = non_zero_indices[changes[0 + 1]]

        # Get the starting and ending time of the direction change
        trade_start_time = signal.index[first_change_start]
        if first_change_end != None:
            trade_end_time = signal.index[first_change_end]
        else:
            trade_end_time = None

        # Return the start and end time tuple
        return (trade_start_time, trade_end_time)
    
    # For index name
    index_name = data.index.name

    # For timeframe
    time_frame = pd.infer_freq(df_for_freq_inferring.index)

    # If timeframe is like 'H' or 'D' or 'Y' then append 1 for specificity
    if len(time_frame) == 1:
        time_frame = '1' + time_frame

    # Extract necessary columns as numpy arrays
    high_prices_1m = data_1m['High']
    low_prices_1m = data_1m['Low']
    open_prices = data['Open']
    signals = data['Signal']
    
    # Initialize the exit indices for tp or sl hit (takes the lowest: which happened first)
    exit_index = None
    
    # Initialize the backtest results array
    backtest_data = []

    # Initialize trade parameters
    tp = 0.05  # 5% take profit
    sl = 0.03  # 3% stop loss
    
    # Initialize the balance
    balance = initial_balance
    
    # Get last date
    last_date = signals.index[-1]
    
    # Initializing List To Store Directions For The Trade
    directions = []
    
    # Iterate
    while(True):
        trade_start_time, trade_end_time = find_first_change(signals)

        if trade_end_time == None:
            break
            
        direction_start = 'long' if signals[trade_start_time] == 1 else 'short'
        direction_end = 'long' if signals[trade_end_time] == 1 else 'short'
        entry_price = open_prices[trade_start_time]
        
        # Calculate take profit and stop loss prices
        if direction_start == 'long':
            tp_price = entry_price * (1 + tp)
            sl_price = entry_price * (1 - sl)
        else:
            tp_price = entry_price * (1 - tp)
            sl_price = entry_price * (1 + sl)
        
        # Find the exit point for the trade
        exit_index = None
        action = 'direction'  # Default action is direction change
        
        # getting to the closest time of that interval
        # Assuming trade_end_time is a datetime object
        # Basically doing this, so the tp and sl hit only checks and
        # compares from the (T + 1)th time till the trade end time.
        trade_start_time_matching = pd.to_datetime(trade_start_time)
        add_minute = pd.Timedelta('1m')
        trade_start_time_matching = trade_start_time_matching + add_minute
        
        if direction_start == 'long':
            tp_hit = np.where(high_prices_1m[trade_start_time_matching:trade_end_time] >= tp_price)[0]
            sl_hit = np.where(low_prices_1m[trade_start_time_matching:trade_end_time] <= sl_price)[0]
        else:
            tp_hit = np.where(low_prices_1m[trade_start_time_matching:trade_end_time] <= tp_price)[0]
            sl_hit = np.where(high_prices_1m[trade_start_time_matching:trade_end_time] >= sl_price)[0]

        if len(tp_hit) > 0:
            exit_index = tp_hit[0]
            action = 'tp'
        if len(sl_hit) > 0 and (len(tp_hit) == 0 or sl_hit[0] < tp_hit[0]):
            exit_index = sl_hit[0]
            action = 'sl'
            
        if action == 'direction':
            close_price = open_prices[trade_end_time]
        else:
            if action == 'tp':
                if direction_start == 'long':
                    trade_end_time = high_prices_1m[trade_start_time_matching:trade_end_time].index[exit_index]
                    close_price = high_prices_1m[trade_end_time]
                else:
                    trade_end_time = low_prices_1m[trade_start_time_matching:trade_end_time].index[exit_index]
                    close_price = low_prices_1m[trade_end_time]
            else:
                if direction_start == 'long':
                    trade_end_time = low_prices_1m[trade_start_time_matching:trade_end_time].index[exit_index]
                    close_price = low_prices_1m[trade_end_time]
                else:
                    trade_end_time = high_prices_1m[trade_start_time_matching:trade_end_time].index[exit_index]
                    close_price = high_prices_1m[trade_end_time]
                    
            if len(directions) != 0:
                direction_end = directions[-1]
                direction_start = directions[-1]
        
        # Record the trade entry and exit
        backtest_data.append([trade_start_time, direction_start, entry_price, 0, None])
        backtest_data.append([trade_end_time, direction_end, entry_price, close_price, action])
            
        # getting to the closest time of that interval
        # Assuming trade_end_time is a datetime object
        trade_end_time = pd.to_datetime(trade_end_time)

        # Define the time format
        time_format = pd.Timedelta(time_frame)

        # Calculate the remainder when trade_end_time is divided by time_format
        remainder = trade_end_time.to_numpy().astype('datetime64[ns]').astype(np.int64) % time_format.to_numpy().astype('timedelta64[ns]').astype(np.int64)

        # If remainder is not zero, round up to the next multiple of time_format
        if remainder != 0:
            trade_end_time = trade_end_time + (time_format - pd.Timedelta(remainder, unit='ns'))
            
        # This is the condition that would end the loop (else it would run infinitely)
        if trade_end_time >= last_date:
            break
            
        signals = signals[trade_end_time:]
        
        directions.append(direction_start)
        directions.append(direction_end)

    # If no signal change detected in the entire signal array, return empty dataframe (for error handling)
    if not backtest_data:
        return pd.DataFrame()

    backtest_df = pd.DataFrame(backtest_data, columns=[index_name, 'direction', 'entry price', 'close price', 'action'])

    # Calculate PNL using vectorized operations
    entry_prices = backtest_df['entry price'][1::2].values
    close_prices = backtest_df['close price'][1::2].values
    directions = backtest_df['direction'][0:-1:2].values

    pnl = np.where(directions == 'long',
                   ((close_prices - entry_prices) / entry_prices) * 100,
                   ((entry_prices - close_prices) / entry_prices) * 100)

    # Insert the PNL values back into the DataFrame
    backtest_df.loc[1::2, 'PNL'] = pnl

    # Update balance considering PNL and transaction fees
    balances = [balance]
    for pnl_value in pnl:
        transaction_cost = balances[-1] * (transaction_fee / 100)
        new_balance = balances[-1] + (np.abs(balances[-1]) * (pnl_value / 100)) - transaction_cost
        balances.append(new_balance)
    
    # Insert the balance values back into the DataFrame
    backtest_df['Balance'] = pd.Series(np.repeat(balances[1:], 2)[:len(backtest_df)])

    # Setting the date as the index of the dataframe
    backtest_df.set_index(index_name, inplace = True)
    backtest_df.index = pd.to_datetime(backtest_df.index, format='mixed')

    return backtest_df

# Calculate all statistics
def calculate_all_statistics(strat_ledger):
    # Get the name of the date column from dataframe
    date_column = strat_ledger.columns[0]

    # Adding the pnl_sum column to the dataframe
    strat_ledger['pnl_sum'] = strat_ledger['PNL'].cumsum()
    
    # Calculate drawdown
    drawdown_list = statistics.calculate_drawdown(strat_ledger['pnl_sum'])
    
    # Calculate PnL sums for different periods
    pnl_sum_scores = statistics.get_last_pnl_scores(strat_ledger)
    
    # Calculate the total PnL percent, total positive and total negative pnl percent as well
    # total_pnl_percent = strat_ledger['PNL'].sum()
    total_pnl_percent, total_neg_pnl_percent, total_pos_pnl_percent = statistics.pos_neg_pnl_percent(strat_ledger['PNL'])

    # Calculate win/loss statistics
    total_wins, total_losses, consecutive_wins, consecutive_losses, win_percentage, loss_percentage = statistics.calculate_wins_losses(strat_ledger)
    win_loss_ratio = statistics.calculate_win_loss_ratio(win_percentage, loss_percentage)
    
    # Calculate average daily PnL
    current_pnl_sum = strat_ledger['pnl_sum'].iloc[-1]
    date_started = pd.to_datetime(strat_ledger[date_column].iloc[0])
    avg_daily_pnl = statistics.average_daily_pnl(current_pnl_sum, date_started)
    
    # Filter dataframe for non-zero close price
    temp_df = strat_ledger[strat_ledger['close price'] != 0]
    
    # Calculate alpha and beta
    alpha, beta = statistics.calculate_alpha_beta(temp_df)
    
    # Calculate Sharpe and Sortino ratios
    sharpe = statistics.calculate_sharpe(temp_df['PNL'])
    sortino = statistics.calculate_sortino(temp_df['PNL'])

    # Calculate r2 score
    r2_score = statistics.calculate_r2_score(strat_ledger)
    
    # Calculate downside risk
    downside_risk = statistics.calculate_downside_risk(temp_df['PNL'])
    
    # Calculate drawdown statistics
    drawdown_durations, max_drawdown, max_drawdown_duration, curr_drawdown, curr_drawdown_duration = statistics.longest_drawdown(
        strat_ledger['pnl_sum'], strat_ledger[date_column]
    )
    average_drawdown = round(np.mean(drawdown_list), 2) if drawdown_list else 0
    average_drawdown_duration = round(np.mean(drawdown_durations), 2) if drawdown_durations else 0
    
    # Create a dictionary with descriptive keys
    stats_dict = {
        date_column: strat_ledger[date_column].iloc[-1],
        'Current Drawdown': -round(float(abs(curr_drawdown)), 2),
        'Current Drawdown Duration (days)': round(float(curr_drawdown_duration), 2),
        'Average Drawdown': -round(float(abs(average_drawdown)), 2),
        'Average Drawdown Duration (days)': round(float(average_drawdown_duration), 2),
        'Maximum Drawdown': -round(float(abs(max_drawdown)), 2),
        'Maximum Drawdown Duration (days)': round(float(max_drawdown_duration), 2),
        'R-squared Score': round(float(r2_score), 2),
        'Sharpe Ratio': round(float(sharpe), 2),
        'Sortino Ratio': round(float(sortino), 2),
        'Total PnL (%)': round(float(total_pnl_percent), 2),
        'Total Positive PnL (%)': round(float(total_pos_pnl_percent), 2),
        'Total Negative PnL (%)': round(float(total_neg_pnl_percent), 2),
        'Total Wins': round(float(total_wins), 2),
        'Total Losses': round(float(total_losses), 2),
        'Consecutive Wins': round(float(consecutive_wins), 2),
        'Consecutive Losses': round(float(consecutive_losses), 2),
        'Win Percentage (%)': round(float(win_percentage), 2),
        'Loss Percentage (%)': round(float(loss_percentage), 2),
        'PnL Sum 1': round(float(pnl_sum_scores[0]), 2),
        'PnL Sum 7': round(float(pnl_sum_scores[1]), 2),
        'PnL Sum 15': round(float(pnl_sum_scores[2]), 2),
        'PnL Sum 30': round(float(pnl_sum_scores[3]), 2),
        'PnL Sum 45': round(float(pnl_sum_scores[4]), 2),
        'PnL Sum 60': round(float(pnl_sum_scores[5]), 2),
        'Average Daily PnL': round(float(avg_daily_pnl), 2),
        'Win/Loss Ratio': round(float(win_loss_ratio), 2),
        'Alpha': round(float(alpha), 2),
        'Beta': round(float(beta), 2),
        'Downside Risk': round(float(downside_risk), 2),
    }

    # Convert dictionary to DataFrame for better visualization (optional)
    stats_df = pd.DataFrame([stats_dict])

    stats_df.set_index(date_column, inplace = True)

    # Convert the index to datetime format
    stats_df.index = pd.to_datetime(stats_df.index)
    
    print(f'{utils.colors.GREEN}All stats calculated!{utils.colors.RESET}')

    return stats_dict, stats_df

def generate_signals_ti(df, strategy):
    match strategy:
        case "MACD":
            print(f"\n{utils.colors.BLUE}MACD Strategy detected.{utils.colors.RESET}")
            
            # Calculate MACD for the 4-Hour data
            signal_df = indicators.calculate_macd(copy.deepcopy(df))
            
            # Generate trading signals based on MACD
            signal_df = indicators.generate_macd_signals(signal_df)
        
        case "ADX_PSAR":
            print(f"\n{utils.colors.BLUE}ADX PSAR Strategy detected.{utils.colors.RESET}")

            # Calculate ADX and Parabolic SAR
            signal_df = indicators.calculate_adx(copy.deepcopy(df), window = 14)
            signal_df = indicators.calculate_parabolic_sar(signal_df)
            
            # Generate trading signals
            signal_df = indicators.generate_adx_parabolic_sar_signals(signal_df)
        
        case "RSI":
            print(f"\n{utils.colors.BLUE}RSI Strategy detected.{utils.colors.RESET}")

            # Calculate RSI for the 1-Hour data
            signal_df = indicators.calculate_rsi(copy.deepcopy(df))
            
            # Generate trading signals based on RSI
            signal_df = indicators.generate_rsi_signals(signal_df)
        
        case "STOCHRSI":
            print(f"\n{utils.colors.BLUE}STOCHASTIC RSI Strategy detected.{utils.colors.RESET}")

            # Calculate RSI for the 1-Hour data
            signal_df = indicators.calculate_stochrsi(copy.deepcopy(df), fillna = True)
            
            # Generate trading signals based on RSI
            signal_df = indicators.generate_stochrsi_signals(signal_df)
        
        case "OBV":
            print(f"\n{utils.colors.BLUE}On-Balance Volume Strategy detected.{utils.colors.RESET}")

            # Calculate The On-Balance Volume (OBV)
            signal_df = indicators.calculate_obv(copy.deepcopy(df), fillna = True)
            
            # Generate Signals Using OBV
            signal_df = indicators.generate_obv_signals(signal_df)

        case "SMA":
            print(f"\n{utils.colors.BLUE}SMA Strategy detected.{utils.colors.RESET}")

            # Assuming `df` is your DataFrame containing 'Close' prices
            signal_df = indicators.generate_sma_signals(copy.deepcopy(df), short_window=50, long_window=200)
        
        case "EMA":
            print(f"\n{utils.colors.BLUE}EMA Strategy detected.{utils.colors.RESET}")

            # Generate trading signals using ema
            signal_df = indicators.generate_triple_ema_signals(copy.deepcopy(df), short_window = 5, medium_window = 21, long_window = 50)
        
        case "SMMA":
            print(f"\n{utils.colors.BLUE}SMMA Strategy detected.{utils.colors.RESET}")

            # Generate trading signals using smma 
            signal_df = indicators.generate_smma_signals(copy.deepcopy(df), window=14)

        case "VWMA":
            print(f"\n{utils.colors.BLUE}VWMA Strategy detected.{utils.colors.RESET}")

            # Generate trading signals using vwma
            signal_df = indicators.generate_vwma_signals(copy.deepcopy(df), short_window=10, long_window=50)
        
        case "AO":
            print(f"\n{utils.colors.BLUE}AWESOME OSCILLATOR Strategy detected.{utils.colors.RESET}")

            # Calculate The Awesome Oscillator (AO)
            signal_df = indicators.calculate_AO(copy.deepcopy(df), fillna = True)
            
            # Generate Signals Using OBV
            signal_df = indicators.generate_AO_signals(signal_df)
            
        case _:
            raise AttributeError(f'\n{utils.colors.WARNING}An Unknown Strategy Name Was Found!{utils.colors.RESET}')

    # Reserve the last signal and prediction time (for the future prediction)
    last_signal = signal_df['Signal'].iloc[-1]
    last_signal_time = signal_df.index[-1]

    # Shift all the signal data points one step ahead (down) to mimic future prediction of past data
    signal_df['Signal'] = signal_df['Signal'].shift(1).fillna(0)
    signal_df = signal_df[1:].copy()

    return signal_df, last_signal, last_signal_time

def generate_signals_ml(df, strategy, model_path):

    # Define the function to generate signals
    def generate_signal(row):
        if row['Close'] > row['Predicted']:
            return -1
        elif row['Close'] < row['Predicted']:
            return 1
        else:
            return 0

    def load_model(model_path):
        """
        Load a model from disk.
        
        Parameters:
        filename (str): The path from which to load the model.
        
        Returns:
        model (sklearn.base.BaseEstimator): The loaded model.
        """
        try:
            model = joblib.load(model_path)
            print(f"{utils.colors.GREEN}Model loaded from {model_path}{utils.colors.RESET}")
            return model
        except Exception as e:
            print(f"{utils.colors.WARNING}Failed to load model. Error: {e}{utils.colors.RESET}")
            return None

    print(f"\n{utils.colors.BLUE}{strategy} Strategy detected.{utils.colors.RESET}")

    # Load the desired model
    model = load_model(model_path)
    if model is None:
        print(f'{utils.colors.WARNING}Unexpcted behaviour. Model loaded as None Type.{utils.colors.RESET}')
        return None, None, None

    # Get the close price predictions from the model
    close_pred = model.predict(df)

    # Add the predictions to the Dataframe
    df['Predicted'] = close_pred

    # Apply the function to create the 'Signal' column
    # and also extract the last predicted signal and
    # the last predicted signal's time for further 
    # use in the simulation code.
    df.loc[:, 'Signal'] = df.apply(generate_signal, axis=1)
    last_signal = df['Signal'].iloc[-1]
    last_signal_time = df.index[-1]

    # Shift the 'Signal' column
    df.loc[:, 'Signal'] = df['Signal'].shift(1).fillna(0)
    
    # Drop the first row because it has a NaN value in 'Signal Actual'
    signal_df = df[1:].copy()

    return signal_df, last_signal, last_signal_time


def simulation(df, df_info):
    
    # Define the global directories
    global current_status_dir
    global pickle_files_dir
    global statistics_dir
    global ledger_dir

    # Store all timeframe dataframes required in a dictionary
    # Assuming df contains your model information
    timeframes = df_info['Timeframe'].unique()  # Extract unique timeframes
    
    # Dictionary to store resampled dataframes
    timeframe_dfs = {}
    
    # Iterate over each unique timeframe and resample the data accordingly
    for timeframe in timeframes:
        # Convert the original DataFrame to the specific timeframe using your conversion function
        resampled_df = utils.convert_1m_to_any_timeframe(copy.deepcopy(df), timeframe)
        
        # Store the resampled dataframe in the dictionary with the timeframe as the key
        timeframe_dfs[timeframe] = resampled_df
    
    # Iterate through each row of the DataFrame
    for _, row in df_info.iterrows():
        strategy = row['Strategy']
        timeframe = row['Timeframe']
        nickname = row['Nickname']
        backbone = row['Model Backbone']
        coin = row['Coin']
        
        # Define the path to all the file
        ledger_file = os.path.join(ledger_dir, f'{nickname}.csv')
        status_file = os.path.join(current_status_dir, f'{nickname}_status.csv')
        statistics_file = os.path.join(statistics_dir, f'{nickname}_stats.csv')
        model_pickle_file = os.path.join(pickle_files_dir, f'{nickname}.pkl')
        
        # Select the correct timeframe DataFrame from the dictionary
        required_df = timeframe_dfs[timeframe]
        
        # Generate signals based on the strategy
        if backbone == 'TI':
            signals_df, last_signal, last_signal_time = generate_signals_ti(copy.deepcopy(required_df), strategy)
        elif backbone == 'ML':
            signals_df, last_signal, last_signal_time = generate_signals_ml(copy.deepcopy(required_df), strategy, model_pickle_file)
        elif backbone == 'DL':
            print(f'{utils.colors.WARNING}Deep Learning Models are not implemented yet. Skipping...{utils.colors.RESET}')
            continue
        elif backbone == 'DARTS':
            print(f'{utils.colors.WARNING}DART Models are not implemented yet. Skipping...{utils.colors.RESET}')
            continue

        # Check if signals_df is None. (Meaning model loaded as None Type: Unexpected Behaviour)
        if signals_df is None:
            continue
        
        # Check if signals_df is not empty before proceeding
        if signals_df.empty:
            print(f"{utils.colors.WARNING}No signals generated for {nickname}. Skipping...{utils.colors.RESET}")
            continue

        # Saving part of signal dataframe for frequency inferring in backtesting function
        df_for_freq_inferring = signals_df[-10:].copy()
        
        # Generate the new ledger using the backtesting function
        if os.path.exists(ledger_file):
            # Reading the exisitng ledger
            existing_ledger = pd.read_csv(ledger_file)
            existing_ledger.set_index(existing_ledger.columns[0], inplace = True)
            existing_ledger.index = pd.to_datetime(existing_ledger.index)

            # Extracting the balance till the last trade
            last_balance_at_that_time = existing_ledger['Balance'].iloc[-1] 

            # Getting only the required signals
            signals_df = signals_df[signals_df.index >= existing_ledger.index[-1]]

            # Generate the new backtest ledger
            new_ledger = generate_backtest_df(signals_df, df[str(signals_df.index[0]): str(signals_df.index[-1])].copy(), df_for_freq_inferring, last_balance_at_that_time)

            # If ledger returned is empty, dont concatenate, else concatenate
            if new_ledger.empty:
                for_stats = existing_ledger.copy()
            else:
                for_stats = pd.concat([existing_ledger, new_ledger])
                for_stats = for_stats.sort_index()
        else:
            new_ledger = generate_backtest_df(signals_df, df[str(signals_df.index[0]): str(signals_df.index[-1])].copy(), df_for_freq_inferring)
            for_stats = new_ledger.copy()

        # If no signal change is detected, (no trade has begun)
        if for_stats.empty:
            print(f"{utils.colors.WARNING}Empty ledger created for {nickname} - no trades held. Skipping...{utils.colors.RESET}")
            continue
        
        # Generate the stats df
        new_ledger_with_all_pnl = statistics.calculate_pnl_sum_all(for_stats.reset_index())
        _, stats_df = calculate_all_statistics(new_ledger_with_all_pnl)

        # Retrieve the cumulative pnl from stats_df to save to metadata csv
        cumulative_pnl = stats_df['Total PnL (%)'].iloc[-1]

        # SAVING TO CSVS / DATABASES
        # Appending or saving to the ledger file
        try:
            # Determine if the file already exists
            if os.path.exists(ledger_file):

                # Check if ledger is empty, then data is already up to date
                if new_ledger.empty:
                    print(f'{utils.colors.YELLOW}Ledger data already up to date!{utils.colors.RESET}')
                else:
                    # Append the newly fetched data to the existing CSV without header
                    new_ledger.to_csv(ledger_file, mode='a', header=False, index=True)
                    print(f"{utils.colors.GREEN}Ledger appended to {utils.colors.BLUE}{ledger_file}{utils.colors.RESET}")
            else:
                # Write the new data with header since file does not exist
                new_ledger.to_csv(ledger_file, mode='w', header=True, index=True)
                print(f"{utils.colors.GREEN}Ledger created and ledger saved to {utils.colors.BLUE}{ledger_file}{utils.colors.RESET}")
            
        except Exception as e:
            print(f"{utils.colors.WARNING}Failed to save / append ledger to {utils.colors.BLUE}{ledger_file}. Error: {e}{utils.colors.RESET}")

        # Saving to the status file
        try:
            # Get the current prediction time
            current_prediction_time = last_signal_time + pd.Timedelta(timeframe)
            
            # Calculate the next prediction time by adding the timeframe as a Timedelta
            next_prediction_time = current_prediction_time + pd.Timedelta(timeframe)
            
            # Convert both times to strings
            current_prediction_time = str(current_prediction_time)
            next_prediction_time = str(next_prediction_time)

            # Get the current prediction
            current_prediction = last_signal

            if new_ledger.empty:
                current_balance = existing_ledger['Balance'].iloc[-1]
            else:
                current_balance = new_ledger['Balance'].iloc[-1]

            pd.DataFrame({
                'Model Name': [nickname],
                'Timeframe' : [timeframe],
                'Current Prediction Time': [current_prediction_time],
                'Current Prediction': [current_prediction],
                'Next Prediction Time': [next_prediction_time],
            }).to_csv(status_file, mode='w', header=True, index=True)

            print(f"{utils.colors.GREEN}Status saved to {utils.colors.BLUE}{status_file}{utils.colors.RESET}")

        except Exception as e:
            print(f"{utils.colors.WARNING}Failed to save status to {utils.colors.BLUE}{status_file}. Error: {e}{utils.colors.RESET}")

        # Saving to the statistics file
        try:
            # Determine if the file already exists
            if os.path.exists(statistics_file):
                # If it exists, read it into a DataFrame
                existing_stats = pd.read_csv(statistics_file)
                existing_stats.set_index(existing_stats.columns[0], inplace = True)
                existing_stats.index = pd.to_datetime(existing_stats.index, format='mixed')
                
                # Find the index where the new ledger starts
                last_stats_index = existing_stats.index[-1]

                # Get only those values that are 
                stats_df = stats_df[stats_df.index > pd.to_datetime(last_stats_index)]

                if stats_df.empty:
                    print(f'{utils.colors.YELLOW}Stats data already up to date!{utils.colors.RESET}{utils.colors.GREEN}')
                else:
                    # Append the newly fetched data to the existing CSV without header
                    stats_df.to_csv(statistics_file, mode='a', header=False, index=True)
                    print(f"{utils.colors.GREEN}Stats appended to {utils.colors.BLUE}{statistics_file}{utils.colors.RESET}")
            else:
                # Write the new data with header since file does not exist
                stats_df.to_csv(statistics_file, mode='w', header=True, index=True)
                print(f"{utils.colors.GREEN}Stats created and stats saved to {utils.colors.BLUE}{statistics_file}{utils.colors.RESET}")
            
        except Exception as e:
            print(f"{utils.colors.WARNING}Failed to save / append stats to {utils.colors.BLUE}{statistics_file}. Error: {e}{utils.colors.RESET}")

        # Saving to the metadata file
        try:
            # Make a list of the new values
            new_values = [nickname, backbone, coin, timeframe, current_prediction_time, current_prediction, next_prediction_time, cumulative_pnl]
            
            # Determine if the file already exists
            if os.path.exists(metadata_file):
                # If it exists, read it into a DataFrame
                existing_metadata = pd.read_csv(metadata_file)
        
                # Check if the model is present and update the row
                if nickname in existing_metadata['Model Name'].values:
                    # Get the current row
                    current_row = existing_metadata.loc[existing_metadata['Model Name'] == nickname]
                    
                    # Check if any of the values are different
                    if current_row.iloc[0].to_list() != new_values:
                        # Update the row with the new values
                        existing_metadata.loc[existing_metadata['Model Name'] == nickname, :] = new_values
                        existing_metadata.to_csv(metadata_file, mode='w', header=True, index=False)
                        print(f"{utils.colors.GREEN}Metadata for {nickname} updated and saved to {utils.colors.BLUE}{metadata_file}{utils.colors.RESET}")
                    else:
                        print(f"{utils.colors.YELLOW}Metadata for {nickname} is already up to date!{utils.colors.RESET}")
                else:
                    # Add a new row to the DataFrame
                    new_row = pd.DataFrame(
                        [new_values],
                        columns = [
                            'Model Name',
                            'Backbone',
                            'Coin',
                            'Timeframe',
                            'Current Prediction Time',
                            'Current Prediction',
                            'Next Prediction Time',
                            'Total PNL'
                        ]
                    )
                    
                    # Add the new row to the existing DataFrame
                    existing_metadata = pd.concat([existing_metadata, new_row], axis=0, ignore_index=True)
        
                    # Overwrite to the new CSV
                    existing_metadata.to_csv(metadata_file, mode='w', header=True, index=False)
                    print(f"{utils.colors.GREEN}Metadata for {nickname} added / appended to {utils.colors.BLUE}{metadata_file}{utils.colors.RESET}")
            else:
                # Write the new data with header since the file does not exist
                pd.DataFrame({
                    'Model Name': [nickname],
                    'Backbone' : [backbone],
                    'Coin': [coin],
                    'Timeframe': [timeframe],
                    'Current Prediction Time': [current_prediction_time],
                    'Current Prediction': [current_prediction],
                    'Next Prediction Time': [next_prediction_time],
                    'Total PNL': [cumulative_pnl]
                }).to_csv(metadata_file, mode='w', header=True, index=False)
                
                print(f"{utils.colors.GREEN}Metadata created and metadata for {nickname} saved to {utils.colors.BLUE}{metadata_file}{utils.colors.RESET}")

        except Exception as e:
            print(f"{utils.colors.WARNING}Failed to save / add / append metadata to {utils.colors.BLUE}{metadata_file}. Error: {e}{utils.colors.RESET}")

        # Adding a new line after every model prediction
        print()

    print(f'\n{utils.colors.GREEN}Simulation Completed Succesfully!{utils.colors.RESET}')


################################################################## Main
def job():
    """
    Job to be executed: Reads the latest data, processes it, and appends the results to a CSV file.
    """
    os.system("")
    global full_csv_path_1
    global full_csv_path_2
    try:
         # Get the current time
        current_time = datetime.datetime.now()

        # Read the model info data
        model_info = pd.read_csv(full_csv_path_2, index_col = 0)

        # Filter the DataFrame based on the current time
        filtered_df = utils.filter_dataframe_by_time(copy.deepcopy(model_info), current_time)

        # If no models fit the time frame, just skip it.and wait for the next hour
        if filtered_df.empty:
            print('No models fit in this timeframe')
        else:
            # Print what timeframe models will be running in the simulation right now
            print(f"{utils.colors.BLUE}\nThe simulation will run on these timeframe models{utils.colors.RESET}")
            print(filtered_df['Timeframe'].unique())

            print(f"{utils.colors.GREEN}\nRunning Simulation!{utils.colors.RESET}")

            # Read some last rows of the ohlc 1 minute data
            ohlc_df = pd.read_csv(utils.get_csv_tail(full_csv_path_1, max_rows=85165), usecols = ['Open time (1M)', 'Open', 'High', 'Low', 'Close', 'Volume'])
            ohlc_df.set_index('Open time (1M)', inplace = True)

            # Convert the index to datetime format
            ohlc_df.index = pd.to_datetime(ohlc_df.index)

            # Run the simulation
            simulation(copy.deepcopy(ohlc_df), filtered_df)

            print("\nSimulation Completed!")

    except Exception as e:
        print(f"Error in simulation: {e}")

# Function to calculate the next execution time with a 5-minute shift on the hour
def get_next_execution_time(current_time):
    """
    Calculate the next execution time that is on the hour with a 10-minute shift.
    
    Args:
    - current_time (datetime): The current time.
    
    Returns:
    - next_time (datetime): The next execution time.
    """
    # Set the next execution time to the next hour with a 5-minute shift
    next_time = current_time.replace(minute=10, second=0, microsecond=0)
    
    # If the current time is already past the calculated next_time, move to the next hour
    if next_time <= current_time:
        next_time += datetime.timedelta(hours=1)
    
    return next_time

# Function to wait until the next execution time
def wait_until_next_execution():
    """
    Wait until the next execution time that is a multiple of 1 hour with a 10-minute shift.
    
    Returns:
    - None
    """
    current_time = datetime.datetime.now()
    next_time = get_next_execution_time(current_time)
    wait_seconds = (next_time - current_time).total_seconds()
    
    if wait_seconds < 0:
        # If wait_seconds is negative, it means the next_time is in the past
        # Calculate the next execution time again
        next_time = get_next_execution_time(datetime.datetime.now())
        wait_seconds = (next_time - datetime.datetime.now()).total_seconds()

    print(f"Waiting for {wait_seconds} seconds until next execution at {next_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    time.sleep(wait_seconds)


######################################################################################## Main
# Main function to run the script continuously
def main():
    """
    Main function to run the technical analysis repeatedly at 5-minute intervals with a 1-minute & 10 second shift.

    This function continuously performs the following steps:
    1. Executes the job function to perform technical analysis.
    2. Waits until the next execution time that is a multiple of 5 minutes with a 1-minute & 10 second shift.

    Returns:
        None
    """
    while True:
        job()   
        wait_until_next_execution()


######################################################################################## Run The Script
if __name__ == "__main__":
    main()