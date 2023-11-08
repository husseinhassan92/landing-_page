#!/usr/bin/python3
"""module for creating models"""

import pandas as pd
import requests
from config import settings
from application.data import AlphaVantageAPI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
import numpy as np


av = AlphaVantageAPI()


class ArimaModelBuilder:
    def __init__(self,ticker="", n_observations=10000):
        self.ticker = ticker
        self.n_observations = n_observations

    def get_data(self, ticker="", n_observations=10000):
        df = av.get_daily(ticker)
        if self.n_observations <= len(df):
            self.new_df = df.iloc[:n_observations, :]
        else:
            self.new_df = df
        self.new_df = self.new_df.asfreq('d')

    def parameters(self):
        """define the parameters of the model"""
        self.y = self.new_df["close"].fillna(method = "ffill")
        self.df_log = self.y
        model_autoARIMA = auto_arima(self.df_log, start_p=0,
                                     start_q=0, start_P=0,
                                     start_Q=0, test='adf', error_action='trace')
        self.get_parametes = model_autoARIMA.get_params()
        summary = model_autoARIMA.summary()
        fig = model_autoARIMA.plot_diagnostics(figsize=(15,8))
        return summary, fig

    def split_data(self,train_size=0.995):
        """split the data to train test sets"""
        limit = int(len(self.df_log) * train_size)
        self.y_train = self.df_log.iloc[:limit]
        self.y_test = self.df_log.iloc[limit:]
        len_train = len(self.y_train)
        len_test = len(self.y_test)
        return len_train, len_test


    def predict(self):
        """"""
        self.order_aa = self.get_parametes.get('order')
        self.model_arima = ARIMA(self.y_train,
                                 order = (self.order_aa[0], self.order_aa[1], self.order_aa[2]))
        self.result = self.model_arima.fit()
        self.y_pred_wfv = self.result.get_forecast(len(self.y_test)+10)
        self.predicted = self.y_pred_wfv.predicted_mean
        self.lower = self.y_pred_wfv.conf_int(0.05).iloc[:, 0]
        self.upper = self.y_pred_wfv.conf_int(0.05).iloc[:, 1]
        self.df_predictions = pd.DataFrame({"train" : self.y_train, "test" : self.y_test, "predict" : self.predicted})
        fig = go.Figure()
        fig.add_trace(go.Line(x=self.y_train.index,y=self.y_train))
        fig.add_trace(go.Line(x=self.y_test.index,y=self.y_test))
        fig.add_trace(go.Line(x=self.predicted.index,y=self.predicted))
        fig.add_trace(go.Line(x=self.lower.index, y=self.lower))
        fig.add_trace(go.Line(x=self.upper.index, y=self.upper,fill='tonexty'))
        return fig

    def forecast(self):
        """graph for the predict prices"""
        self.y_pred_wfv = pd.Series()
        self.history = self.y_train.copy()
        for i in range(len(self.y_test)):
            self.model = ARIMA(self.history, order = (self.order_aa[0], self.order_aa[1], self.order_aa[2])).fit()
            self.next_pred = self.model.forecast()
            self.y_pred_wfv = self.y_pred_wfv._append(self.next_pred)
            self.history = self.history._append(self.y_test[self.next_pred.index])
        self.df_predictions = pd.DataFrame({"train" : self.y_train, "y_test" : self.y_test,"y_pred" : self.y_pred_wfv})
        fig = px.line(self.df_predictions)
        return fig