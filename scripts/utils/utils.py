################################################################ Libraries
# For Manipulating Date
import datetime

# Other Necessary Libraries
import pandas as pd
import tailer
import io


################################################################## Functions and Classes
class colors:
    BLUE    = '\033[94m'
    YELLOW  = '\033[33m'
    GREEN   = '\033[92m'
    WARNING = '\033[31m'
    RESET   = '\033[0m'

# Function to return the timeframe of the dataframe's index
def get_timeframe(df):
    """
    Returns the inferred timeframe (frequency) of the datetime index of a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with datetime index.
    
    Returns:
        str: Timeframe (frequency) of the datetime index, e.g., '4H', '1D'.
    """
    # Check if the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime.")
    
    # Infer the frequency of the datetime index
    frequency = pd.infer_freq(df.index)

    if len(frequency) == 1:
        frequency = '1' + frequency
    
    if frequency is None:
        raise ValueError("Unable to infer the frequency of the datetime index.")
    
    return frequency

# Function that extracts previous 'N' times data from the latest data
def get_past_data(df, period, timeframe):
    """
    Returns data from the past specified period and aligns the start and end dates with the given timeframe.
    
    Parameters:
        df (pd.DataFrame): DataFrame with datetime index.
        period (str): Period string, e.g., '1Y' for one year, '1M' for one month.
        timeframe (str): Timeframe string, e.g., '1M' for one month, '4H' for four hours, '15T' for fifteen minutes.
    
    Returns:
        pd.DataFrame: DataFrame filtered by the specified period and aligned with the timeframe.
    """
    # Check if the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime.")
    
    # Mapping period strings to DateOffset arguments
    period_mapping = {
        'Y': 'years',
        'M': 'months',
        'W': 'weeks',
        'D': 'days',
        'H': 'hours',
        'T': 'minutes'
    }
    
    # Extract the time unit and quantity for the period
    period_unit = period[-1]
    period_quantity = int(period[:-1])
    
    # Get the corresponding DateOffset argument for the period
    if period_unit not in period_mapping:
        raise ValueError("Invalid period format. Use formats like '1Y', '1M', '1W', '1D', etc.")
    
    period_offset_arg = period_mapping[period_unit]
    period_offset = pd.DateOffset(**{period_offset_arg: period_quantity})
    
    # Get the last date in the DataFrame
    last_date = df.index[-1]
    
    # Calculate the start date
    start_date = last_date - period_offset
    
    # Align start_date to the nearest preceding datetime divisible by the timeframe
    freq = pd.tseries.frequencies.to_offset(timeframe)
    aligned_start_date = start_date.floor(freq)
    
    # Align end_date to the nearest preceding datetime divisible by the timeframe
    aligned_end_date = last_date.floor(freq)
    
    # Filter the DataFrame for the desired date range
    filtered_df_last = df[(df.index >= aligned_start_date) & (df.index <= aligned_end_date)]
    filtered_df_first = df[(df.index < aligned_start_date)]
    
    return filtered_df_first, filtered_df_last

# Function that extracts the last n/2 rows of the csv
def get_csv_tail(filepath, max_rows=1):
    with open(filepath) as file:
        # Read the header
        header = file.readline().strip()
        
        # Read the last lines of the file
        last_lines = tailer.tail(file, max_rows)
        last_lines = last_lines[1:]
        
    # Combine the header with the last lines
    combined_lines = '\n'.join([header] + last_lines)

    return io.StringIO(combined_lines)

# Functions converts the dataframe into any given time frame.
def convert_1m_to_any_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert a DataFrame of 1-minute OHLC data to any given timeframe.

    Args:
    - df (pd.DataFrame): DataFrame containing 1-minute OHLC data. 
      The DataFrame should have a datetime index & columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    - timeframe (str): The desired timeframe to resample the data to (e.g., '1H' for 1 hour, '1D' for 1 day).

    Returns:
    - pd.DataFrame: Resampled DataFrame with OHLC data in the specified timeframe. The index will be renamed to
      reflect the new timeframe.

    Example:
    ```
    resampled_df = convert_1m_to_any_timeframe(ohlc_df, '1H')
    ```
    """
    # Ensure the DataFrame index is of datetime type
    df.index = pd.to_datetime(df.index)
    
    # Try resampling the data to the desired timeframe
    try:
        df_resampled = df.resample(timeframe).agg({
            'Open': 'first',  # Take the first 'Open' value in the timeframe
            'High': 'max',    # Take the maximum 'High' value in the timeframe
            'Low': 'min',     # Take the minimum 'Low' value in the timeframe
            'Close': 'last',  # Take the last 'Close' value in the timeframe
            'Volume': 'mean'  # Take the mean 'Volume' value in the timeframe
        })
    except Exception as e:
        print(f"An error occurred while resampling! Error message: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    # Rename the index to reflect the new timeframe
    df_resampled.index.rename(f'Open time ({timeframe})', inplace = True)
    
    return df_resampled

# Function that returns the number of seconds corresponding to the timeframe
def get_seconds_from_timeframe(timeframe):
    """
    Convert timeframe string (e.g., '1h', '4h') to seconds.
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == 'h':
        return value * 3600  # Convert hours to seconds
    elif unit == 'm':
        return value * 60    # Convert minutes to seconds
    elif unit == 's':
        return value         # Already in seconds
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")

# Functon that filters the models_info dataframe corresponding to the timeframe
def filter_dataframe_by_time(df, current_time):
    # Convert the current time to seconds since midnight
    seconds_since_midnight = current_time.hour * 3600 + current_time.minute * 60 - 600 # Subtracting 5 minutes

    # Filter the dataframe based on whether the current time is divisible by the timeframe in seconds
    filtered_df = df[df['Timeframe'].apply(lambda tf: seconds_since_midnight % get_seconds_from_timeframe(tf) == 0)]

    return filtered_df

# Function to change the fetched data into a dataframe
def ohlc_to_dataframe(data: list) -> pd.DataFrame:
    """
    Convert OHLC (Open, High, Low, Close) data to a pandas DataFrame with correct column names and index.

    Args:
    - data (list of lists): List containing OHLC data where each element is a list representing a row of OHLC data.

    Returns:
    - df (pd.DataFrame): DataFrame with OHLC data, indexed by 'Open time (1M)' & columns for 'Open', 'High', 'Low',
      'Close', 'Volume'.
    """
    # Define column names based on Binance API response
    columns = [
        'Open time (1M)', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close time', 'Quote asset volume', 'Number of trades', 
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    
    # Create DataFrame from data with specified column names
    df = pd.DataFrame(data, columns = columns)
    
    # Convert 'Open time (1M)' to datetime & set as index
    df['Open time (1M)'] = pd.to_datetime(df['Open time (1M)'], unit = 'ms')
    df.set_index('Open time (1M)', inplace = True)
    
    # Drop unnecessary columns
    df.drop(columns = ['Close time', 'Ignore'], inplace = True)
    
    # Convert all columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors = 'coerce')
    
    return df

# Function to return the latest timestamp from the ohlc data
def get_latest_timestamp(csv_full_path):
    """
    Retrieve the latest timestamp from the last row of a CSV file.

    Args:
        csv_filename (str): Name of the CSV file to read from.

    Returns:
        str: The latest timestamp as a string in the format "%Y-%m-%d %H:%M:%S".
    
    Raises:
        Exception: If the timestamp cannot be found or extracted from the file.
    """
    try:
        # Open the CSV file and read the last line to get the latest timestamp
        with open(csv_full_path, 'r') as f:
            last_row = f.readlines()[-1]

            # Extract the latest timestamp from the last row
            latest_timestamp = ''
            for ch in last_row:
                if ch == ',':
                    break
                latest_timestamp = latest_timestamp + ch

            if last_row:
                latest_timestamp = datetime.datetime.strptime(latest_timestamp, "%Y-%m-%d %H:%M:%S")
                start_str = (latest_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            else:
                raise Exception('Unexpected Error: Time not found or extracted!')

        return start_str

    except Exception as e:
        print(f"Failed to read {csv_full_path}. Error: {e}")
        return