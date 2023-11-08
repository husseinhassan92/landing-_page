#!/usr/bin/python3
"""module for getting the data from Api"""

import pandas as pd
import requests
from config import settings


class AlphaVantageAPI:
    def __init__(self, api_key=settings.api_key):
        self.__api_key=api_key

    def get_symbol(self, keywords):
        """Get the best match of the key woards.

        Parameters
        ----------
        keywords : str
            The ticker symbol of the equity.

        Returns
        -------
        dict file
            key : symbol,
            value : name of company
        """
        self.keywords = keywords

        url = (
            "https://www.alphavantage.co/query?"
            "function=SYMBOL_SEARCH"
            f"&keywords={self.keywords}"
            f"&apikey={settings.api_key}"
            )
        response=requests.get(url=url)
        response_data=response.json()
        bestMatches=response_data.get('bestMatches')
        search = {}
        for i in range(len(bestMatches)):
            search[str((bestMatches[i].get('1. symbol'), bestMatches[i].get('2. name')))] = bestMatches[i].get('1. symbol')
        return search

    def get_daily(self, ticker, output_size = "full"):

        """Get daily time series of an equity from AlphaVantage API.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the equity.
        output_size : str, optional
            Number of observations to retrieve. "compact" returns the
            latest 100 observations. "full" returns all observations for
            equity. By default "full".

        Returns
        -------
        pd.DataFrame
            Columns are 'open', 'high', 'low', 'close', and 'volume'.
            All are numeric.
        """
        # Create URL (8.1.5)
        ticker=ticker
        output_size=output_size
        data_type="json"

        url = (
            "https://www.alphavantage.co/query?"
            "function=TIME_SERIES_DAILY"
            f"&symbol={ticker}"
            f"&outputsize={output_size}"
            f"&datatype={data_type}"
            f"&apikey={self.__api_key}"
        )

        # Send request to API (8.1.6)
        response=requests.get(url=url)

        # Extract JSON data from response (8.1.10)
        response_data=response.json()

        if "Time Series (Daily)" not in response_data.keys():
            raise Exception(
                f"Invalid Api call. check the ticker '{ticker}' is correct"
            )

        # Read data into DataFrame (8.1.12 & 8.1.13)
        stock_data=response_data["Time Series (Daily)"]
        df=pd.DataFrame.from_dict(stock_data, orient = "index", dtype = float)

        # Convert index to `DatetimeIndex` named "date" (8.1.14)
        df.index=pd.to_datetime(df.index)
        df.index.name="date"

        # Remove numbering from columns (8.1.15)
        df.columns=[c.split(" ")[1] for c in df.columns]
        return df

    def get_overview(self, ticker):
        """Get overview info about company from AlphaVantage API.
        Parameters
        ----------
        ticker : str
        The ticker symbol of the equity.
        
        
        Returns
        -------
        dict
        info about the company
        """
        
        ticker=ticker
        
        url = (
            "https://www.alphavantage.co/query?"
            "function=OVERVIEW"
            f"&symbol={ticker}"
            f"&apikey={settings.api_key}"
            )
        
        response=requests.get(url=url)
        response_data=response.json()
        return response_data