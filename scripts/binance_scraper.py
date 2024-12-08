####################################################################################### All Libraries
# Binance Package
from binance.futures import Futures

# For Manipulating Date
import datetime
import time

# Library To Parse The Config File
import configparser

# Other Necessary Libraries
import pandas as pd
import os

# Libraries to Make Code Cleaner
from typing import List

# For Network Checking
import socket

# Importing Custom Functions
from utils import utils


####################################################################################### All Declarations
# Settig Up The Directory From The Directory Hierarchy
# Get the script's current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the parent directory
parent_dir = os.path.dirname(script_dir)

# define path to all the needed directories within 'parent' and 'data' directory
config_dir = os.path.join(script_dir, "configuration")
data_dir = os.path.join(parent_dir, "data")
ohlc_dir = os.path.join(data_dir, "ohlc")

# Create directories if they do not exist
os.makedirs(config_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(ohlc_dir, exist_ok=True)

# Create the full path to the config file within the 'configuration' directory relative to this path
config_file = os.path.join(config_dir, "config.ini")

# Read configuration settings
config = configparser.ConfigParser()
config.read(config_file)

# Defining my API key & my secret key as well
API_Secret = config['SCRAPPER']['API_Secret']
API_Key = config['SCRAPPER']['API_Key']

# File name for the data to be appended to
csv_filename = config['SCRAPPER']['ohlc_data']

# Define the interval
interval = config['SCRAPPER']['interval']

# The coin that you need the data for
symbol = config['SCRAPPER']['symbol']

# The base url through which we will scrape our data
base_url = config['SCRAPPER']['base_url']

# Creat the full path to the ohlc data in the 'ohlc' directory within the 'data' directory
csv_full_path = os.path.join(ohlc_dir, csv_filename)


######################################################################################## Client Initializing
# Call the constructor of futures
client = Futures()
client = Futures(key = API_Key, secret = API_Secret, base_url = base_url)


######################################################################################## All Functions
# Function to check internet connectivity
def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if the internet connection is available.

    Args:
    - host (str): The host to connect to. Default is Google's public DNS server.
    - port (int): The port to connect to. Default is 53 (DNS service port).
    - timeout (int): Timeout duration in seconds. Default is 3.

    Returns:
    - bool: True if the internet connection is available, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(f"{utils.colors.WARNING}No internet connection. Error: {ex}{utils.colors.RESET}")
        return False

# Function to get & return OHLC data for a given COIN (symbol)
def fetch_ohlc_data(symbol: str, interval: str, start_str: str) -> List[List[float]]:
    """
    Fetch historical klines (candlestick) data from Binance Futures API.

    Args:
    - symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
    - interval (str): The interval of the kline (e.g., '1m' for 1 minute, '1h' for 1 hour).
    - start_str (str): The start date & time in the format "%Y-%m-%d %H:%M:%S".

    Returns:
    - klines (List[List[float]]): A list of kline data where each element is a list representing a kline.
    """
    klines = []
    limit  = 1000  # Maximum number of records per request

    # Convert start_str to datetime objects
    start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    end_dt   = datetime.datetime.now()

    # Loop until we fetch all required data from start_dt to end_dt
    while start_dt < end_dt:
        start_ts = int(start_dt.timestamp() * 1000)

        # Fetch klines data from Binance API
        temp_klines = client.klines(
            symbol    = symbol,
            interval  = interval,
            startTime = start_ts,
            limit     = limit
        )

        # Check if no more data is available
        if not temp_klines:
            break
        
        # Append fetched klines to the result list
        klines = klines + temp_klines
        
        # Update start_dt to the next timestamp after the last fetched kline
        start_dt = datetime.datetime.fromtimestamp(temp_klines[-1][0] / 1000.0) + datetime.timedelta(minutes = 1)
        
        # Break loop if fetched data is less than the limit
        if len(temp_klines) < limit:
            break
        
        # Sleep to avoid hitting rate limits
        time.sleep(1)

    return klines

# Function to append new data to the CSV
def append_to_csv(new_data: pd.DataFrame, csv_full_path: str):
    """
    Append new data to the existing CSV file in the 'ohlc' folder within the 'data' directory.
    Creates 'data' and 'ohlc' directories if they do not exist.

    Args:
    - new_data (pd.DataFrame): DataFrame containing new data to append.
    - filename (str): Name of the CSV file to append to.

    Returns:
    - None
    """
    try:
        # Determine if the file already exists
        if os.path.exists(csv_full_path):
            # Append the newly fetched data to the existing CSV without header
            new_data.to_csv(csv_full_path, mode='a', header=False, index=True)
        else:
            # Write the new data with header since file does not exist
            new_data.to_csv(csv_full_path, mode='w', header=True, index=True)
        
        print(f"{utils.colors.GREEN}Data appended to {csv_full_path}{utils.colors.RESET}")

    except Exception as e:
        print(f"{utils.colors.WARNING}Failed to append data to {csv_full_path}. Error: {e}{utils.colors.RESET}")

# Function to shift time (5 hours ahead) and save to csv
def shift_time_and_save_to_csv(df, csv_full_path, shift_amount):
    """
    Shift the index of a DataFrame by a specified amount and append the data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be shifted and saved.
        shift_amount (str): The amount of time to shift the DataFrame index, e.g., '5H' for 5 hours.
    
    Returns:
        None
    """
    # Convert shift_amount to a Timedelta and shift the DataFrame index
    shift_amount = pd.Timedelta(shift_amount)
    df.index = df.index.map(lambda x: x + shift_amount)
            
    # Drop the last row to avoid duplicates and anomalies
    if not df.empty:
        df = df.iloc[1:]
    
    # Append new data to the CSV
    if not df.empty:
        append_to_csv(df, csv_full_path)

def job():
    """
    Job to be executed: Reads the latest data, processes it, and appends the results to a CSV file.
    """
    global csv_full_path

    try:
        print(f"{utils.colors.BLUE}\nFetching Data!{utils.colors.RESET}")

        # Fetching the start time
        start_str = utils.get_latest_timestamp(csv_full_path)
        print(f"{utils.colors.BLUE}Start time is: {start_str}{utils.colors.RESET}")
        
        # Fetch new data from Binance
        new_ohlc_data = fetch_ohlc_data(symbol, interval, start_str)
        
        # Convert new data to DataFrame
        new_df = utils.ohlc_to_dataframe(new_ohlc_data)

        # I dont know why, but needed to shift time by 5 hours..
        shift_time_and_save_to_csv(new_df, csv_full_path, '5h')

        print(f"{utils.colors.GREEN}Data Fetched!{utils.colors.RESET}")

    except Exception as e:
        print(f"{utils.colors.WARNING}Error in fetching data: {e}{utils.colors.RESET}")

# Function to calculate next execution time with a 1-minute 10-second shift
def get_next_execution_time(current_time):
    """
    Calculate the next execution time that is a multiple of 5 minutes with a 1-minute 10-second shift.
    
    Args:
    - current_time (datetime): The current time.
    
    Returns:
    - next_time (datetime): The next execution time.
    """
    next_time = current_time.replace(second=10, microsecond=0) + datetime.timedelta(minutes=(6 - (current_time.minute % 5)))
    
    # If the current time is already past the calculated next_time, move to the next interval
    if next_time <= current_time:
        next_time += datetime.timedelta(minutes=5)

    return next_time

# Function to wait until the next execution time
def wait_until_next_execution():
    """
    Wait until the next execution time that is a multiple of 5 minutes with a 1-minute 10-second shift.
    
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

    print(f"{utils.colors.BLUE}Waiting for {wait_seconds} seconds until next execution at {next_time.strftime('%Y-%m-%d %H:%M:%S')}...{utils.colors.RESET}")
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
        os.system("")
        # Check internet connection before running the job
        if check_internet_connection():
            # Run the job
            job()
            wait_until_next_execution()
        else:
            print(f"{utils.colors.YELLOW}Waiting for internet connection...{utils.colors.RESET}")
            time.sleep(10)  # Wait 10 seconds before checking again


######################################################################################## Run The Script
if __name__ == "__main__":
    main()