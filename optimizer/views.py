##################################################
################ SCRIPT PACKAGES #################
import pandas as pd
import numpy as np

import yfinance as yf
import datetime as dt

from pypfopt import risk_models, expected_returns, objective_functions, HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR
from pypfopt.discrete_allocation import DiscreteAllocation

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################
################ DJANGO STUFF #################

from django.shortcuts import render
from django.views import View
from .forms import InvestmentForm
from .functions import Optimizer, Backtester, Plotter
from django.conf import settings

from django.core.cache import cache
###############################################

# Load the dataset
data_df = cache.get('data_df')
if data_df is None:
   data_df = pd.read_csv("optimizer/df3.csv")
   cache.set('data_df', data_df, timeout=None)  # Cache indefinitely


class Index(View):
    def get(self, request):
        form = InvestmentForm()
        return render(request, 'optimizer/index.html', {'form': form})
        
    def post(self, request):
        form = InvestmentForm(request.POST)

        if form.is_valid():
            try:
                stocks_input = form.cleaned_data['stocks'].strip().upper()
                
                # Check if the input is empty or does not contain any comma
                if not stocks_input or ',' not in stocks_input:
                    error_msg = "Please enter at least two stock tickers separated by commas."
                    return render(request, 'optimizer/index.html', {'form': form, 'error_msg': error_msg})
                
                # Split the input by comma
                stocks = [ticker.strip() for ticker in stocks_input.split(',') if ticker.strip()]
                
                # Check if all tickers are valid (e.g., only alphabetic characters)
                not_available = [ticker for ticker in stocks if ticker.upper() not in data_df['ticker'].str.upper().values]
                if not_available:
                    error_msg = f"Stock(s): {', '.join(not_available)} not available in the dataset. Make sure you enter the correct ticker(s)."
                    return render(request, 'optimizer/index.html', {'form': form, 'error_msg': error_msg})
                
                # To remove duplicates
                stocks = list(set(stocks))

                # Check if stocks are available in the dataset and say which one(s) is/are not available
                not_available = [ticker for ticker in stocks if ticker not in data_df['ticker'].values]
                if not_available:
                    error_msg = f"Stock(s): {', '.join(not_available)} not available in the dataset."
                    return render(request, 'optimizer/index.html', {'form': form, 'error_msg': error_msg})

                # Continue with your optimization process
                stocks = form.cleaned_data['stocks']
                optimization_method = form.cleaned_data['optimization_method']
                risk_aversion = int(form.cleaned_data['risk_aversion'])
                risk_free = float(form.cleaned_data['risk_free']) / 100
                rebalance_freq = int(form.cleaned_data['rebalance_freq'])
                money = int(form.cleaned_data['money'])
                benchmark = form.cleaned_data['benchmark']

                ## Initialize the Optimizer class with the user's inputs.
                optimizer = Optimizer(stocks=stocks, optimization_method=optimization_method, risk_aversion=risk_aversion, risk_free=risk_free, money=money)
                data = optimizer.get_data()

                mu = optimizer.get_mu(data)
                covmat = optimizer.get_covmat(data)

                log_ret = optimizer.get_logret(data)

                ## Initialize the Backtester class with the user's inputs.
                backtest = Backtester(stocks, optimization_method=optimization_method, lookback=optimizer.get_lookback(data), rebalance_freq=rebalance_freq, money=money,
                                        risk_aversion=risk_aversion, risk_free=risk_free, benchmark=benchmark)

                subframes = backtest.expanding_window(data)
                weights_df = backtest.optimize_back(subframes)

                if weights_df is None:
                    error_msg = "Error occurred during optimization. Try again or adjust your parameters."
                    return render(request, 'optimizer/index.html', {'form': form, 'error_msg': error_msg})

                alphas_list = backtest.alphas_list(data, weights_df)
                alphas_df = backtest.alphas_df(alphas_list, log_ret)

                expost_perf = backtest.expost_perf(alphas_df, log_ret)

                last_weights = np.round(weights_df.iloc[:,-1],8)
                discrete = optimizer.discrete_values(data, OrderedDict(last_weights))

                benchmark_ret = backtest.benchmark_data(expost_perf, benchmark)

                ## Initialize the Plotter class with the user's inputs.
                plotter = Plotter()

                pie_plot = plotter.pie_plot(last_weights)
                line_plot = plotter.portfolio_vs_benchmark(expost_perf, benchmark_ret)
                efficient_plot = plotter.plot_efficient_frontier(mu, covmat, weights_df)
                metrics_table = plotter.metrics_table(expost_perf, benchmark_ret, risk_free, money)
                recap_table = plotter.portfolio_alloc_recap(last_weights, discrete, data_df)

                context = {
                    'pie_plot': pie_plot,
                    'line_plot': line_plot,
                    'efficient_plot': efficient_plot,
                    'metrics_table': metrics_table.to_html(index=False),
                    'recap_table': recap_table.to_html(index=False),
                }

                return render(request, 'optimizer/result.html', context)
            except Exception as e:
                error_msg = "Error during the optimization. Try again with different parameters."
                return render(request, 'optimizer/index.html', {'form': form, 'error_msg': error_msg})
        else:
            # Form is not valid, render the form again with errors
            return render(request, 'optimizer/index.html', {'form': form})
