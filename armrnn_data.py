import yfinance as yf
from datetime import datetime
import pandas as pd


def get_armnn_data(tickers, start, end, filename, p=4, k=1):
    """
    Align data for Autogressive Moving Reference Neural Network.
    (R(t-(p-1)) - z ...... R(t) - z) -->> (R(t+1) - z) where z = R(t-(p-1)-k)

    In the main case, 4 lagged daily returns will be used to predict tomorrow's return
    All returns are normalized by substracting the 5th lagged return (z)
    ARMNN(p=4, k=1):
    (R(t-3) - z, R(t-2) - z, .... ,R(t) - z) -->> (R(t+1) -z) where z = R(t-4)

    Pandas Dataframe including Y and X variables will be saved as filename.csv
    """

    prices = yf.download(tickers=tickers, start=start, end=end)['Adj Close']
    armrnn_data = pd.DataFrame()
    for stock in prices.columns:
        returns = prices[stock].pct_change()

        temp = pd.DataFrame()
        z = returns.shift(p+k)
        temp['TP1'] = returns - z

        for i in range(p):
            temp[f'TM{i}'] = returns.shift(i+1) - z
        temp = temp.dropna().reset_index(drop=True)

        armrnn_data = pd.concat([armrnn_data, temp])
    print('Written to CSV:')
    print(armrnn_data)
    armrnn_data.to_csv(f'{filename}.csv', index=False)


if __name__ == '__main__':
    tickers = ['UNH', 'MSFT', 'GS', 'HD', 'MCD',
               'CAT', 'AMGN', 'V', 'BA', 'CRM',
               'HON', 'AAPL', 'TRV', 'AXP', 'JNJ',
               'CVX', 'WMT', 'PG', 'JPM', 'IBM',
               'NKE', 'MRK', 'MMM', 'DIS', 'KO',
               'DOW', 'CSCO', 'VZ', 'INTC', 'WBA']
    start = datetime(2012, 1, 1)
    end = datetime(2023, 1, 1)

    get_armnn_data(tickers, start=start, end=end, filename='test', p=4, k=1)

