# Crypbot: A Trading Bot

Crypbot is an advanced trading bot that leverages machine learning algorithms to predict Bitcoin's closing price with high accuracy. By analyzing historical OHLC data, it dynamically adjusts trading strategies to optimize profits and minimize risks.

## Features

- **Automated Trading**: Automatically opens and closes trades based on real-time signals.
- **Technical Indicators**: Utilizes various technical indicators such as SMA, EMA, SMMA, and VWMA.
- **Signal Generation**: Generates buy/sell signals based on crossover strategies.
- **Trade Management**: Manages open positions and calculates PnL.
- **Historical Data Analysis**: Analyzes historical data to improve prediction accuracy.


## Getting Started

### Prerequisites

- Python 3.8+
- Binance API Key and Secret

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/iabahmad/Cryp-bot.git
    cd Crypbot
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Configure your Binance API credentials in `scripts/configuration/config.ini`.

### Usage

1. Start the trading bot:
    ```sh
    python app.py
    ```

2. Monitor the bot's performance and logs in `trading.log`.

### Running Simulations

1. Run the simulation script:
    ```sh
    python scripts/simulation.py
    ```

### Technical Indicators

Crypbot uses various technical indicators to generate trading signals. Some of the key indicators include:

- **Simple Moving Average (SMA)**: [calculate_sma](scripts/technical_indicators/indicators.py)
- **Exponential Moving Average (EMA)**: [calculate_ema](scripts/technical_indicators/indicators.py)
- **Smoothed Moving Average (SMMA)**: [calculate_smma](scripts/technical_indicators/indicators.py)
- **Volume Weighted Moving Average (VWMA)**: [calculate_vwma](scripts/technical_indicators/indicators.py)

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- [Binance API](https://github.com/binance/binance-spot-api-docs)
- [Pandas](https://pandas.pydata.org/)
- [TA-Lib](https://mrjbq7.github.io/ta-lib/)

---

Happy Trading!