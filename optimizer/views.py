##################################################
################ SCRIPT PACKAGES #################
import pandas as pd
import numpy as np

import yfinance as yf
import datetime as dt

from pypfopt import risk_models, expected_returns, objective_functions, HRPOpt, plotting
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
###############################################

class Index(View):
    def get(self, request):

        form = InvestmentForm()
        return render(request, 'optimizer/index.html', {'form': form})
        
    def post(self, request):
        form = InvestmentForm(request.POST)

        if form.is_valid():
            stocks = form.cleaned_data['stocks'].split(", ")
            optimization_method = form.cleaned_data['optimization_method']
            lookback = int(form.cleaned_data['lookback'])
            risk_aversion = int(form.cleaned_data['risk_aversion'])
            risk_free = float(form.cleaned_data['risk_free']) / 100
            rebalance_freq = int(form.cleaned_data['rebalance_freq'])
            money = int(form.cleaned_data['money'])
            benchmark = form.cleaned_data['benchmark']

            ## Initialize the Optimizer class with the user's inputs.
            optimizer = Optimizer(stocks=stocks, optimization_method=optimization_method, lookback=lookback, risk_aversion=risk_aversion, risk_free=risk_free, money=money)
            data = optimizer.get_data()
            log_ret = optimizer.get_logret(data)

            mu = optimizer.get_mu(data)
            covmat = optimizer.get_covmat(data)

            weights = optimizer.optimize(data, covmat, mu)
            discrete = optimizer.discrete_values(data, weights)


            ## Initialize the Backtester class with the user's inputs.
            backtest = Backtester(optimization_method=optimization_method, lookback=lookback, rebalance_freq=rebalance_freq, money=money,
                                  risk_aversion=risk_aversion, risk_free=risk_free, benchmark=benchmark)

            subframes = backtest.expanding_window(data)
            weights_df = backtest.optimize_back(subframes)

            alphas_list = backtest.alphas_list(data, weights_df)
            alphas_df = backtest.alphas_df(alphas_list, log_ret)

            expost_perf = backtest.expost_perf(alphas_df, log_ret)
            benchmark_ret = backtest.benchmark_data(data, benchmark)

            ## Initialize the Plotter class with the user's inputs.
            plotter = Plotter()

            pie_plot = plotter.pie_plot(weights_df)
            line_plot = plotter.portfolio_vs_benchmark(expost_perf, benchmark_ret)
            efficient_plot = plotter.plot_efficient_frontier(mu, covmat, weights_df)

            portfolio_table = plotter.portfolio_metrics(expost_perf, risk_free, money)
            benchmark_table = plotter.benchmark_metrics(expost_perf, benchmark_ret, risk_free, money)
            metrics_table = pd.merge(portfolio_table, benchmark_table, right_on=benchmark_table.index, left_on=portfolio_table.index)
            metrics_table.rename(columns={'key_0': 'performance metrics'}, inplace=True)
            metrics_table.set_index('performance metrics', inplace=True)

            recap_table = plotter.portfolio_alloc_recap(weights_df, discrete)

            context = {
                'pie_plot': pie_plot,
                'line_plot': line_plot,
                'efficient_plot': efficient_plot,
                'metrics_table': metrics_table.reset_index().to_html(index=False),
                'recap_table': recap_table.reset_index().to_html(index=False)
,
            }

        return render(request, 'optimizer/result.html', context)



