import requests
import json
import requests_cache
from typing import Union
import pandas as pd
from datetime import datetime
import numpy as np

class RequestProcessor:
    def __init__(self, base_url, cache_name='api_cache', expire_after=3600):
        self.base_url = base_url
        requests_cache.install_cache(cache_name, expire_after=expire_after) # create a global cache for all requests. To lower the number of calls made to the server side

    def get_data(self, start_date: Union[datetime, str], end_date: Union[datetime, str], product_name: str, metrics: str, get_mocked_data=False):

        if get_mocked_data:
            with open('sample_response.json', 'r') as f:
                data = json.loads(f.read())
            return self.__to_dataframe(data)
        
        start_date = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else end_date


        url = f"{self.base_url}/data"
        body = {
            'start_date': start_date,
            'end_date': end_date,
            'product_name': product_name,
            'metrics': metrics
        }
        
        try:
            response = requests.post(url, json=body, verify=True)
            response.raise_for_status()
            data = response.json()
            return self.__to_dataframe(data)
        except requests.exceptions.SSLError as e:
            print(f"SSL error occurred: {e}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()

    def __to_dataframe(self, data):
        try:
            df = pd.DataFrame(data['result'])
            return df
        except ValueError as e:
            print(f"Error converting data to DataFrame: {e}")
            return pd.DataFrame()

    def get_implied_rate_gbp_usd(self ,eurusd, eurgbp):
        return eurusd['price'] / eurgbp['price']
    
    def get_rolling_std_dev(self, data):
        """
        Returns the rolling standard deviation of the price over a 10 day window
        """
        return data['price'].rolling(window=10).std()
    
    def return_change_in_price(self, data, S=0):
        """
        Returns the change in price from the previous day
        """
        return data['price'].diff()
    
    def cumm_pl(self, data, S=0):
        """
        Returns cumulative sum of the price
        """
        if S == 0:
            return data['price'].cumsum()
        return data['price'][:S].cumsum()
    
    def get_drawdown(self, df):
        """
        Returns the drawdown of the price
        """
        df['PL'] = df['price'].diff()
        df['CumulativePL'] = df['PL'].cumsum()
        df['CumulativePL_max'] = df['CumulativePL'].cummax()
        df['Drawdown'] = df['CumulativePL_max'] - df['CumulativePL']
        max_drawdown_date = df.iloc[df['Drawdown'].idxmax(), 'Date']
        drawdown_start_date = df.iloc[df.loc[:max_drawdown_date, 'CumulativePL'].idxmax(), 'Date']
        return max_drawdown_date, drawdown_start_date
    
    def annualized_sharpe_ratio(df, risk_free_rate=0):
        daily_returns = df['price'].pct_change().dropna()
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio