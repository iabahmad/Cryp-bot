################################################################ Libraries
# For Manipulating Date
import datetime

# Other Necessary Libraries
import quantstats as qs
import pandas as pd
import numpy as np


################################################################## Functions
# Get the last row of all pnl scores
def get_last_pnl_scores(ledger):
    # List of columns to extract
    pnl_cols = ['pnl_sum_1', 'pnl_sum_7', 'pnl_sum_15', 'pnl_sum_30', 'pnl_sum_45', 'pnl_sum_60']
    last_values = ledger[pnl_cols].iloc[-1].values
    
    return last_values.tolist()

# Calculate the 1d, 7d, 15d, 30d, 45d, 60d PNL scores
def calculate_pnl_sum_all(df):
    date_column = df.columns[0]
    
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Ensure the PNL column is numeric and handle any non-numeric values
    df['PNL'] = pd.to_numeric(df['PNL'], errors='coerce').fillna(0.0)
    
    # Adding the cumulative sum column to the dataframe
    df['pnl_sum'] = df['PNL'].cumsum()
    
    # Precompute the time deltas
    time_deltas = {
        'pnl_sum_1': datetime.timedelta(days=1),
        'pnl_sum_7': datetime.timedelta(days=7),
        'pnl_sum_15': datetime.timedelta(days=15),
        'pnl_sum_30': datetime.timedelta(days=30),
        'pnl_sum_45': datetime.timedelta(days=45),
        'pnl_sum_60': datetime.timedelta(days=60)
    }
    
    # Initialize columns with NaN values
    for col_name in time_deltas.keys():
        df[col_name] = np.nan
    
    # Set the date column as the index
    df.set_index(date_column, inplace=True)
    
    for col_name, delta in time_deltas.items():
        window_days = delta.days
        # Calculate the rolling sum with a time-based window
        rolling_sums = df['PNL'].rolling(window=f'{window_days}D', closed='both').sum()
        
        # Align rolling sums with the original DataFrame
        df[col_name] = rolling_sums.reindex(df.index).fillna(0.0)
    
    # Reset index to get date column back
    df.reset_index(inplace=True)
    
    # Round the results to 2 decimal places
    df = df.round({col: 2 for col in time_deltas.keys()})
    
    return df

# Calculate the difference of date
def calculate_diff_date(start, end):
    return (end - start).days

# calculate drawdown longest drawdown,current drawdown
def longest_drawdown(pnl_cum_list, date):
    max_drawdown = 0
    max_drawdown_duration = 0
    curr_drawdown = 0
    curr_drawdown_duration = 0
    drawdown_durations = []

    maxPnl = pnl_cum_list[0]
    start_date = None
    drawdown_active = False

    for counter, value in enumerate(pnl_cum_list):
        if value < maxPnl:
            drawdown = maxPnl - value

            if not drawdown_active:
                start_date = date.iloc[counter]  # Use iloc to access by position
                drawdown_active = True

            curr_drawdown = drawdown
            curr_drawdown_duration = calculate_diff_date(start_date, date.iloc[counter])  # Use iloc

            if curr_drawdown_duration > max_drawdown_duration:
                max_drawdown_duration = curr_drawdown_duration

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        elif drawdown_active:
            end_date = date.iloc[counter]  # Use iloc
            drawdown_durations.append(calculate_diff_date(start_date, end_date))
            drawdown_active = False
            start_date = None  # Reset start_date after the drawdown ends
            curr_drawdown_duration = 0  # Reset current drawdown duration
            maxPnl = value

        if value > maxPnl:
            maxPnl = value

    # Ensure the current drawdown duration is updated correctly
    if drawdown_active:
        curr_drawdown_duration = calculate_diff_date(start_date, date.iloc[-1])  # Use iloc

    return drawdown_durations, round(max_drawdown, 2), max_drawdown_duration, round(curr_drawdown, 2), curr_drawdown_duration

# Calculate drawdown
def calculate_drawdown(pnl_cum_list):
    drawdown_list = []
    maxPnl = pnl_cum_list[0]

    for value in pnl_cum_list:
        maxPnl = max(maxPnl, value)
        drawdown = round(value - maxPnl, 2)
        drawdown_list.append(drawdown)

    return drawdown_list

# win/losses calculation
def calculate_wins_losses(df):
    total_wins = total_losses = consecutive_wins = consecutive_losses = 0
    temp_wins = temp_losses = 0

    for pnl in df['PNL'][1:]:
        if pnl > 0:
            total_wins += 1
            temp_wins += 1
            if temp_losses > consecutive_losses:
                consecutive_losses = temp_losses
            temp_losses = 0
        elif pnl < 0:
            total_losses += 1
            temp_losses += 1
            if temp_wins > consecutive_wins:
                consecutive_wins = temp_wins
            temp_wins = 0

    win_percentage = round(total_wins / (total_wins + total_losses) * 100, 2)
    loss_percentage = round(total_losses / (total_wins + total_losses) * 100, 2)

    return total_wins, total_losses, consecutive_wins, consecutive_losses, win_percentage, loss_percentage
    
# Calculate r2 score
def calculate_r2_score(ledger):
    y = ledger.pnl_sum.to_numpy()
    x = np.arange(len(y))
    
    # Mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Centered variables
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Covariance of x and y
    covariance = np.sum(x_centered * y_centered)
    
    # Variance of x and y
    variance_x = np.sum(x_centered ** 2)
    variance_y = np.sum(y_centered ** 2)
    
    # Calculate the correlation coefficient
    correlation = covariance / np.sqrt(variance_x * variance_y)
    
    # Calculate R^2 score
    r2 = correlation ** 2
    
    return round(r2, 2)

# positive negative pnl calculation
def pos_neg_pnl_percent(pnl_percent):
    # Negative PnL sum directly from filtering
    total_neg_pnl_percent = pnl_percent[pnl_percent < 0].sum()
    # total_neg_pnl_percent = neg_pnl_percent.sum()

    # Positive PnL sum directly from filtering
    total_pos_pnl_percent = pnl_percent[pnl_percent > 0].sum()

    # Total PnL percent (no need to store intermediate results)
    return total_neg_pnl_percent + total_pos_pnl_percent, total_neg_pnl_percent, total_pos_pnl_percent

# sharp calculation
def calculate_sharpe(returns):
    # Calculate the sharpe ratio using QuantStats library with risk free rate = 0 (2nd parameter)
    sharpe_ratio = qs.stats.sharpe(returns, 0)

    return round(sharpe_ratio,2)

# Calculate downside risk
def calculate_downside_risk(returns, risk_free=0):
    # Calculate adjusted returns by subtracting the risk-free rate
    adj_returns = returns - risk_free
    
    # Calculate squared downside risk
    sqr_downside = np.square(np.minimum(adj_returns, 0))
    
    # Calculate the mean of the squared downside and scale it by 252 (annualization factor)
    mean_sqr_downside = np.nanmean(sqr_downside)
    
    # Return the square root of the annualized downside risk
    return np.sqrt(mean_sqr_downside * 252)

# Calculate sortino
def calculate_sortino(returns):
    # Calculate the sortino using QuantStats library
    sortino=qs.stats.sortino(returns)
    
    return sortino

# Calculate average daily pnl
def average_daily_pnl(pnl_sum, date_started):
    # Ensure date_started is in datetime format
    if isinstance(date_started, str):
        date_started = datetime.strptime(date_started, '%Y-%m-%d %H:%M:%S')
    
    # Calculate the number of days between date_started and now
    delta = (datetime.datetime.now() - date_started).days
    
    # Calculate the average daily PnL
    daily_pnl = pnl_sum / delta
    
    return daily_pnl

# Caculate win / loss ratio
def calculate_win_loss_ratio(win_percentage, loss_percentage):
    # Handle division by zero by returning the win_percentage if loss_percentage is 0
    if loss_percentage == 0:
        return win_percentage
    
    # Calculate and return the win/loss ratio
    return win_percentage / loss_percentage

# Calculate alpha beta
def calculate_alpha_beta(df):
    # Convert necessary columns to NumPy arrays for faster computation
    close_price = df['close price'].astype(float).values
    entry_price = df['entry price'].astype(float).values
    pnl = df['PNL'].astype(float).values
    
    # Calculate btc_return using vectorized operations
    btc_return = (close_price / entry_price - 1) * 100
    
    # Linear regression using NumPy
    # Add a constant term (intercept) to the predictor
    X = np.vstack((pnl, np.ones_like(pnl))).T
    Y = btc_return
    
    # Solve for alpha (slope) and beta (intercept)
    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    # Coefficients[0] = alpha, Coefficients[1] = beta
    return coefficients[0], coefficients[1]