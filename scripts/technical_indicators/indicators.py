################################################################ Libraries
# Technical Analysis Library Modules
from ta.momentum import AwesomeOscillatorIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import StochRSIIndicator
from ta.momentum import RSIIndicator
from ta.trend import PSARIndicator
from ta.trend import ADXIndicator
from ta.trend import SMAIndicator
from ta.trend import EMAIndicator

# Other Necessary Libraries
import pandas as pd
import numpy as np


################################################################## Functions
def calculate_macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD (Moving Average Convergence Divergence) for a given DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' price data.
    - short_window (int): The window size for the short-term EMA, default is 12.
    - long_window (int): The window size for the long-term EMA, default is 26.
    - signal_window (int): The window size for the Signal line, default is 9.

    Returns:
    - pd.DataFrame: DataFrame with the MACD line and Signal line.
    """
    # Calculate the short-term and long-term EMAs
    df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    # Calculate the MACD line
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Calculate the Signal line
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

    return df

def generate_macd_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on MACD values.

    Args:
    - df (pd.DataFrame): DataFrame containing 'MACD' and 'Signal_Line' values.

    Returns:
    - pd.DataFrame: DataFrame with trading signals.
    """
    df['Signal'] = 0  # Default no signal
    df.loc[df['MACD'] > df['Signal_Line'], 'Signal'] = 1  # Buy signal
    df.loc[df['MACD'] < df['Signal_Line'], 'Signal'] = -1  # Sell signal

    return df

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX) and add it to the DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
    - window (int): Window size for the ADX calculation (default is 14).
    
    Returns:
    - df (pd.DataFrame): DataFrame with added 'ADX' column.
    """
    adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=window)
    df['ADX'] = adx.adx()
    df['DI+'] = adx.adx_pos()
    df['DI-'] = adx.adx_neg()
    return df

def calculate_parabolic_sar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Parabolic SAR and add it to the DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
    
    Returns:
    - df (pd.DataFrame): DataFrame with added 'Parabolic_SAR' column.
    """
    psar = PSARIndicator(df['High'], df['Low'], df['Close'])
    df['Parabolic_SAR'] = psar.psar()
    return df

def generate_adx_parabolic_sar_signals(df: pd.DataFrame, adx_threshold: float = 25) -> pd.DataFrame:
    """
    Generate trading signals based on ADX and Parabolic SAR.

    Args:
    - df (pd.DataFrame): DataFrame containing 'ADX' and 'Parabolic_SAR' columns.
    - adx_threshold (float): Threshold for ADX to consider a strong trend (default is 25).

    Returns:
    - df (pd.DataFrame): DataFrame with added 'Signal' column (1 for buy, -1 for sell, 0 for hold).
    """
    df['Signal'] = 0  # Default to hold
    df.loc[(df['ADX'] > adx_threshold) & (df['Close'] > df['Parabolic_SAR']), 'Signal'] = 1  # Buy signal
    df.loc[(df['ADX'] > adx_threshold) & (df['Close'] < df['Parabolic_SAR']), 'Signal'] = -1  # Sell signal
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) using the ta library.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' price data.
    - window (int): The window size for calculating RSI, default is 14.

    Returns:
    - pd.DataFrame: DataFrame with the RSI values.
    """
    # Calculate RSI using ta library
    rsi_indicator = RSIIndicator(close=df['Close'], window=window, fillna=True)
    df['RSI'] = rsi_indicator.rsi()

    return df

def generate_rsi_signals(df: pd.DataFrame, rsi_lower: int = 30, rsi_upper: int = 70) -> pd.DataFrame:
    """
    Generate trading signals based on RSI values.

    Args:
    - df (pd.DataFrame): DataFrame containing 'RSI' values.
    - rsi_lower (int): RSI lower threshold for buy signals, default is 30.
    - rsi_upper (int): RSI upper threshold for sell signals, default is 70.

    Returns:
    - pd.DataFrame: DataFrame with trading signals.
    """
    df['Signal'] = 0  # Default no signal
    df.loc[df['RSI'] > rsi_upper, 'Signal'] = -1  # Sell signal
    df.loc[df['RSI'] < rsi_lower, 'Signal'] = 1   # Buy signal

    return df

def calculate_stochrsi(
    df: pd.DataFrame,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Stochastic RSI (STOCHRSI) for a given DataFrame using the ta library.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' price data.
    - window (int): The window size for calculating RSI, default is 14.
    - smooth1 (int): The window size for the first smoothing, default is 3.
    - smooth2 (int): The window size for the second smoothing, default is 3.
    - fillna (bool): The paramter to specify whether to fill NaN values or not.

    Returns:
    - df (pd.DataFrame): DataFrame with the Stochastic RSI values.
    """

    # Calculate Stochastic RSI
    stoch_rsi = StochRSIIndicator(
        close=df['Close'],
        window = window,
        smooth1 = smooth1,
        smooth2 = smooth2,
        fillna = fillna
    )

    # Add Stochastic RSI values to the DataFrame
    df['StochRSI'] = stoch_rsi.stochrsi()
    
    df['StochRSI_K'] = stoch_rsi.stochrsi_k() * 100
    df['StochRSI_D'] = stoch_rsi.stochrsi_d() * 100

    return df

def generate_stochrsi_signals(
    df: pd.DataFrame,
    stochrsi_upper: int = 0.7,
    stochrsi_lower: int = 0.3,
) -> pd.DataFrame:
    """
    Calculate the Stochastic RSI (STOCHRSI) and generate buy/sell signals for a given DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' price data.
    - window (int): The window size for calculating RSI, default is 14.
    - smooth1 (int): The window size for the first smoothing, default is 3.
    - smooth2 (int): The window size for the second smoothing, default is 3.
    - fillna (bool): The parameter to specify whether to fill NaN values or not.

    Returns:
    - df (pd.DataFrame): DataFrame with the Stochastic RSI values and signals.
    """
    # Generate signals
    df['Signal'] = 0
    df['Signal'] = np.where((df['StochRSI'].shift(1) < 0.2) & (df['StochRSI'] >= 0.2), 1, df['Signal'])  # Buy signal
    df['Signal'] = np.where((df['StochRSI'].shift(1) > 0.8) & (df['StochRSI'] <= 0.8), -1, df['Signal'])  # Sell signal

    return df

def calculate_obv(df: pd.DataFrame, fillna: bool = False) -> pd.DataFrame:
    """
    Generate On-Balance Volume (OBV) Values.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.
    - fillna (bool): Parameter that specifies whether or not to fill NaN values.

    Returns:
    - df (pd.DataFrame): DataFrame with added 'OBV' column.
    """
    obv = OnBalanceVolumeIndicator(
        close = df['Close'],
        volume = df['Volume'],
        fillna = True
    )
    
    df['OBV'] = obv.on_balance_volume()

    return df


def generate_obv_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on On-Balance Volume (OBV) indicator.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.
    - fillna (bool): Parameter that specifies whether or not to fill NaN values.

    Returns:
    - df (pd.DataFrame): DataFrame with added 'OBV' and 'Signal' columns.
    """
    df['Signal'] = 0  # Initialize Signal column
    
    # Generate signals based on OBV
    df.loc[df['OBV'] > df['OBV'].shift(1), 'Signal'] = 1  # Buy signal
    df.loc[df['OBV'] < df['OBV'].shift(1), 'Signal'] = -1  # Sell signal

    return df

def calculate_sma(
    df: pd.DataFrame, 
    window: int, 
    source: str = 'Close', 
    offset: int = 0,
    fillna: bool = False
) -> pd.DataFrame:
    """
    Calculate the Simple Moving Average (SMA) for a given DataFrame with additional parameters.

    Args:
    - df (pd.DataFrame): DataFrame containing price data.
    - window (int): The window size for calculating SMA.
    - source (str): The column name on which to calculate the SMA, default is 'Close'.
    - offset (int): The number of periods to offset the SMA, default is 0.

    Returns:
    - df (pd.DataFrame): DataFrame with the SMA values.
    """
    sma_indicator = SMAIndicator(close = df[source], window = window, fillna = fillna)
    df[f'SMA_{window}'] = sma_indicator.sma_indicator()

    if offset != 0:
        df[f'SMA_{window}'] = df[f'SMA_{window}'].shift(offset)
    
    return df

def generate_sma_signals(
    df: pd.DataFrame, 
    short_window: int = 50, 
    long_window: int = 200
) -> pd.DataFrame:
    """
    Generate trading signals based on Simple Moving Average (SMA) crossover strategy.

    Args:
    - df (pd.DataFrame): DataFrame containing price data with SMA values.
    - short_window (int): The window size for the short-term SMA, default is 50.
    - long_window (int): The window size for the long-term SMA, default is 200.

    Returns:
    - pd.DataFrame: DataFrame with trading signals.
    """
    # Calculate short-term and long-term SMAs
    df = calculate_sma(df, window=short_window, source='Close')
    df = calculate_sma(df, window=long_window, source='Close')
    
    # Generate signals: 
    # Buy when the short-term SMA crosses above the long-term SMA
    # Sell when the short-term SMA crosses below the long-term SMA
    df['Signal'] = 0  # Default no signal
    df.loc[df[f'SMA_{short_window}'] > df[f'SMA_{long_window}'], 'Signal'] = 1   # Buy signal
    df.loc[df[f'SMA_{short_window}'] < df[f'SMA_{long_window}'], 'Signal'] = -1  # Sell signal

    return df

def calculate_ema(
    df: pd.DataFrame, 
    window: int, 
    source: str = 'Close', 
    offset: int = 0, 
    fillna: bool = False,
    smoothing_line: str = 'ema', 
    smoothing_length: int = None
) -> pd.DataFrame:
    """
    Calculate the Exponential Moving Average (EMA) for a given DataFrame with additional parameters.

    Args:
    - df (pd.DataFrame): DataFrame containing price data.
    - window (int): The window size for calculating EMA.
    - source (str): The column name on which to calculate the EMA, default is 'Close'.
    - offset (int): The number of periods to offset the EMA, default is 0.
    - smoothing_line (str): The type of smoothing line, default is 'ema' (only EMA supported in this function).
    - smoothing_length (int): The window size for additional smoothing, not used in this function. Will implement later
    - fillna (bool): The parameter to specifiy if NaN values are to be filled or not

    Returns:
    - df (pd.DataFrame): DataFrame with the EMA values.
    """
    if smoothing_line != 'ema':
        raise ValueError("Only 'ema' smoothing is supported in this function.")
    
    ema_indicator = EMAIndicator(close = df[source], window = window, fillna = fillna)
    df[f'EMA_{window}'] = ema_indicator.ema_indicator()

    if offset != 0:
        df[f'EMA_{window}'] = df[f'EMA_{window}'].shift(offset)
    
    return df

def generate_triple_ema_signals(
    df: pd.DataFrame, 
    short_window: int = 5, 
    medium_window: int = 21, 
    long_window: int = 50,
    source: str = 'Close'
) -> pd.DataFrame:
    """
    Generate trading signals based on the Triple EMA Crossover strategy.

    Args:
    - df (pd.DataFrame): DataFrame containing price data.
    - short_window (int): The window size for the short-term EMA.
    - medium_window (int): The window size for the medium-term EMA.
    - long_window (int): The window size for the long-term EMA.
    - source (str): The column name on which to calculate the EMAs, default is 'Close'.

    Returns:
    - pd.DataFrame: DataFrame with trading signals.
    """
    # Calculate short-term, medium-term, and long-term EMAs
    df = calculate_ema(df, short_window, source)
    df = calculate_ema(df, medium_window, source)
    df = calculate_ema(df, long_window, source)

    # Initialize signal column
    df['Signal'] = 0

    # Buy signal: Short EMA crosses above both Medium and Long EMAs
    df.loc[(df[f'EMA_{short_window}'] > df[f'EMA_{medium_window}']) & (df[f'EMA_{short_window}'] > df[f'EMA_{long_window}']), 'Signal'] = 1

    # Sell signal: Short EMA crosses below both Medium and Long EMAs
    df.loc[(df[f'EMA_{short_window}'] < df[f'EMA_{medium_window}']) & (df[f'EMA_{short_window}'] < df[f'EMA_{long_window}']), 'Signal'] = -1

    return df

def calculate_smma(
    df: pd.DataFrame, 
    window: int, 
    source: str = 'Close', 
    offset: int = 0
) -> pd.DataFrame:
    """
    Calculate the Smoothed Moving Average (SMMA) for a given DataFrame with additional parameters.

    Args:
    - df (pd.DataFrame): DataFrame containing price data.
    - window (int): The window size for calculating SMMA.
    - source (str): The column name on which to calculate the SMMA, default is 'Close'.
    - offset (int): The number of periods to offset the SMMA, default is 0.

    Returns:
    - df (pd.DataFrame): DataFrame with the SMMA values.
    """
    # Initialize SMMA column
    df['SMMA'] = 0.0

    # Calculate initial SMMA values (SMA for the first window periods)
    df['SUM1'] = df[source].rolling(window=window, min_periods=1).sum()
    df.loc[df.index[window - 1], 'SMMA'] = df['SUM1'].iloc[window - 1] / window

    # Calculate subsequent SMMA values using the iterative formula
    for i in range(window, len(df)):
        prev_smma = df.loc[df.index[i - 1], 'SMMA']
        current_price = df.loc[df.index[i], source]
        df.loc[df.index[i], 'SMMA'] = (prev_smma * (window - 1) + current_price) / window

    # Drop intermediate columns if not needed
    df.drop(['SUM1'], axis=1, inplace=True)

    return df

def generate_smma_signals(
    df: pd.DataFrame, 
    window: int = 14, 
    source: str = 'Close'
) -> pd.DataFrame:
    """
    Generate trading signals based on the Smoothed Moving Average (SMMA) crossover strategy.

    Args:
    - df (pd.DataFrame): DataFrame containing price data.
    - window (int): The window size for calculating SMMA.
    - source (str): The column name on which to calculate the SMMA, default is 'Close'.

    Returns:
    - pd.DataFrame: DataFrame with trading signals.
    """
    # Calculate SMMA
    df = calculate_smma(df, window, source)

    # Initialize signal column
    df['Signal'] = 0

    # Buy signal: When price crosses above SMMA
    df.loc[df[source] > df['SMMA'], 'Signal'] = 1

    # Sell signal: When price crosses below SMMA
    df.loc[df[source] < df['SMMA'], 'Signal'] = -1

    return df

def calculate_vwma(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Volume Weighted Moving Average (VWMA) for a given DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing price data and volume.
    - window (int): The window size for calculating VWMA.

    Returns:
    - pd.DataFrame: DataFrame with the VWMA values.
    """
    df['PriceVolume'] = df['Close'] * df['Volume']
    df['CumulativePriceVolume'] = df['PriceVolume'].rolling(window=window, min_periods=1).sum()
    df['CumulativeVolume'] = df['Volume'].rolling(window=window, min_periods=1).sum()
    df[f'VWMA_{window}'] = df['CumulativePriceVolume'] / df['CumulativeVolume']

    # Drop intermediate columns if not needed
    df.drop(['PriceVolume', 'CumulativePriceVolume', 'CumulativeVolume'], axis=1, inplace=True)
    
    return df

def generate_vwma_signals(df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.DataFrame:
    """
    Generate trading signals based on VWMA crossover strategy.

    Args:
    - df (pd.DataFrame): DataFrame containing price data and volume.
    - short_window (int): Window size for short-term VWMA.
    - long_window (int): Window size for long-term VWMA.

    Returns:
    - pd.DataFrame: DataFrame with VWMA values and trading signals.
    """
    # Calculate short-term VWMA
    df = calculate_vwma(df, window=short_window)
    df.rename(columns={f'VWMA_{short_window}': 'VWMA_Short'}, inplace=True)

    # Calculate long-term VWMA
    df = calculate_vwma(df, window=long_window)
    df.rename(columns={f'VWMA_{long_window}': 'VWMA_Long'}, inplace=True)

    # Generate trading signals
    df['Signal'] = 0
    df['Signal'] = df.apply(lambda row: 1 if row['VWMA_Short'] > row['VWMA_Long'] else (-1 if row['VWMA_Short'] < row['VWMA_Long'] else 0), axis=1)

    return df

def calculate_AO(df: pd.DataFrame, fillna: bool = False) -> pd.DataFrame:
    """
    Generate On-Balance Volume (OBV) Values.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.
    - fillna (bool): Parameter that specifies whether or not to fill NaN values.

    Returns:
    - df (pd.DataFrame): DataFrame with added 'OBV' column.
    """
    obv = AwesomeOscillatorIndicator(
        high = df['High'],
        low = df['Low'],
        fillna = True
    )
    
    df['AO'] = obv.awesome_oscillator()

    return df


def generate_AO_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on Awesome Oscillator (AO) values.

    Args:
    - df (pd.DataFrame): DataFrame containing 'AO' column. Should have a datetime index.

    Returns:
    - df (pd.DataFrame): DataFrame with added 'Signal' column indicating Buy (1), Sell (-1), or Hold (0) signals.
    """
    # Initialize Signal column with zeros
    df['Signal'] = 0
    
    # Generate signals based on AO
    for i in range(1, len(df)):
        if df['AO'].iloc[i] > 0 and df['AO'].iloc[i-1] <= 0:
            df.at[df.index[i], 'Signal'] = 1  # Buy signal
        elif df['AO'].iloc[i] < 0 and df['AO'].iloc[i-1] >= 0:
            df.at[df.index[i], 'Signal'] = -1  # Sell signal
    
    return df