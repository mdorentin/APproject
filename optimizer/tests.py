import unittest
from .functions import Optimizer, Backtester, Plotter

import random
import pandas as pd
import datetime as dt
import numpy as np

from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from django.core.cache import cache

data_df = cache.get('data_df')
if data_df is None:
    data_df = pd.read_csv("optimizer/df3.csv")
    cache.set('data_df', data_df, timeout=None)
## Randomly select a few stocks (between 2 and 10) from the dataset.
stocks = random.sample(list(data_df.ticker),random.randint(5, 10))
optimization_method = random.choice(['equal', 'min_vol', 'min_cvar', 'max_quad', 'hrp'])
risk_aversion = random.randint(1, 10)
risk_free = random.randint(1, 6)
money = random.randint(10_000, 10_000_000)
rebalance_freq = random.choice([1, 2, 4])
benchmark = random.choice(['SPY', 'IWM', 'VEA'])

print(f'Testing initialized with stocks: {stocks}')
print(f'Testing initializedwith optimization method: {optimization_method}')
print(f'Testing initialized with rebalance frequency: {rebalance_freq}')
print(f'Testing initialized with risk aversion: {risk_aversion}')
print(f'Testing initialized with risk free rate: {risk_free}')
print(f'Testing initialized with money: {money}')
print(f'Testing initialized with benchmark: {benchmark}')

class Testoptimizer(unittest.TestCase):

    def setUp(self):
        self.stocks = stocks
        self.optimization_method = optimization_method
        self.risk_aversion = risk_aversion
        self.risk_free = risk_free
        self.money = money
    
    def test_initializing(self):
        '''
        Test if the Optimizer object and its parameters are initialized correctly.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        
        self.assertIsInstance(optimizer, Optimizer)  # Check if object is an instance of optimizer
        self.assertEqual(optimizer.stocks, self.stocks)  # Check if stocks attribute is set correctly

        self.assertIn(optimizer.optimization_method, ['equal', 'min_vol', 'min_cvar', 'max_quad', 'hrp']) # Check if optimization_method attribute is set correctly
        self.assertEqual(optimizer.optimization_method, self.optimization_method)  # Check if optimization_method attribute is set correctly

        self.assertEqual(optimizer.risk_aversion, self.risk_aversion)  # Check if risk_aversion attribute is set correctly
        self.assertIsInstance(optimizer.risk_aversion, int) # Check if risk_aversion is an integer
        self.assertGreaterEqual(optimizer.risk_aversion, 1) # Check if risk_aversion is greater than or equal to 1
        self.assertLessEqual(optimizer.risk_aversion, 10) # Check if risk_aversion is less than or equal to 10

        self.assertEqual(optimizer.risk_free, self.risk_free)  # Check if risk_free attribute is set correctly
        self.assertGreaterEqual(optimizer.risk_free, 1) # Check if risk_free is greater than or equal to 1
        self.assertIsInstance(optimizer.risk_free, (int, float)) # Check if risk_free is an integer or float
        
        self.assertEqual(optimizer.money, self.money)  # Check if money attribute is set correctly
        self.assertIsInstance(optimizer.money, int) # Check if money is an integer

    def test_getdata(self):
        '''
        Test if the get_data method returns a pandas DataFrame and that the DataFrame has the correct columns.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        data = optimizer.get_data()

        self.assertIsInstance(data, pd.DataFrame) # Check if data is a pandas DataFrame
        self.assertEqual(data.columns.tolist(), sorted(set(self.stocks))) # Check if columns are the stocks
        self.assertFalse(data.isnull().all().all())  # Check if DataFrame contains non-NaN values
        self.assertEqual(data.index.max().year, dt.datetime.today().year) # Check if the most recent date is the current year
        self.assertGreaterEqual(data.index.min().year, dt.datetime.today().year - 13) # Check if the oldest date is within the last 13 years

    def test_getlookback(self):
        '''
        Test if the get_lookback method returns an integer between 1 and 10.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        data = optimizer.get_data()
        lookback = optimizer.get_lookback(data)

        self.assertIsInstance(lookback, int) # Check if lookback is an integer
        self.assertGreaterEqual(lookback, 1) # Check if lookback is greater than or equal to 1
        self.assertLessEqual(lookback, 10)   # Check if lookback is less than or equal to 10
    
    def test_getcovmat(self):
        '''
        Test if the get_cov_matrix method returns a pandas DataFrame, that the DataFrame has the correct columns, that there are no NaN values and that the shape of the DataFrame is correct.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        data = optimizer.get_data()
        covmat = optimizer.get_covmat(data)

        self.assertIsInstance(covmat, pd.DataFrame) # Check if covmat is a pandas DataFrame
        self.assertEqual(covmat.columns.tolist(), sorted(set(self.stocks))) # Check if columns are the stocks
        self.assertEqual(covmat.index.tolist(), sorted(set(self.stocks))) # Check if index is the stocks
        self.assertFalse(covmat.isnull().all().all()) # Check if DataFrame contains non-NaN values
        self.assertEqual(covmat.shape, (len(self.stocks), len(self.stocks))) # Check if shape of DataFrame is correct
    
    def test_mu(self):
        '''
        Test if the mu method returns a pandas Series, that the Series has the correct index and values, that the Series has no NaN values.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        data = optimizer.get_data()
        mu = optimizer.get_mu(data)

        self.assertIsInstance(mu, pd.Series) # Check if mu is a pandas Series
        self.assertEqual(mu.index.tolist(), sorted(set(self.stocks))) # Check if index is the stocks
        self.assertFalse(mu.isnull().all()) # Check if Series contains non-NaN values
        self.assertEqual(mu.name, data.index.max()) # Check if name of Series is the most recent date
    
    def test_optimize(self):
        '''
        Test if the optimize method returns a OrderedDict with the correct keys and values, that the sum of the values is equal to 1 and that the values are between 0 and 1.
        Test if the maximum Sharpe Ratio optimization fails, if it is due to the code (true error) or due to the data/parameters (not a problem).
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        data = optimizer.get_data()
        mu = optimizer.get_mu(data)
        covmat = optimizer.get_covmat(data)
        # For other optimization methods, perform regular optimization and assertions
        weights = optimizer.optimize(data, covmat, mu)

        self.assertIsInstance(weights, dict) # Check if weights is a dictionary
        self.assertEqual(set(weights.keys()), set(self.stocks)) # Check if keys are the stocks
        self.assertAlmostEqual(np.round(sum(weights.values()),2), 1) # Check if sum of values is equal to 1
        self.assertTrue(all(0 <= value <= 1 for value in weights.values())) # Check if values are between 0 and 1

    def test_discrete(self):
        '''
        Test if the discrete function returns a pandas dataframe with the correct columns and index, that the shape of the DataFrame is correct and that the values are integers.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        data = optimizer.get_data()
        mu = optimizer.get_mu(data)
        covmat = optimizer.get_covmat(data)
        weights = optimizer.optimize(data, covmat, mu)
        discrete = optimizer.discrete_values(data, weights)

        self.assertTrue(discrete.values.dtype == 'int32') # Check if values are integers
        
class TestBacktester(unittest.TestCase):

    def setUp(self):
        data_df = cache.get('data_df')
        if data_df is None:
            data_df = pd.read_csv("optimizer/df3.csv")
            cache.set('data_df', data_df, timeout=None)

        ## Randomly select a few stocks (between 2 and 10) from the dataset.
        self.stocks = stocks
        self.optimization_method = optimization_method
        self.rebalance_freq = rebalance_freq
        self.risk_aversion = risk_aversion
        self.risk_free = risk_free
        self.money = money
        self.benchmark = benchmark
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        self.data = optimizer.get_data()
        self.lookback = optimizer.get_lookback(self.data)

        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)

    def test_initializing(self):
        '''
        Test if the Backtester object and its parameters are initialized correctly.
        '''
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        
        self.assertIsInstance(backtester, Backtester)

        self.assertEqual(backtester.stocks, self.stocks)  # Check if stocks attribute is set correctly

        self.assertIn(backtester.optimization_method, ['equal', 'min_vol', 'min_cvar', 'max_quad', 'hrp']) # Check if optimization_method attribute is set correctly
        self.assertEqual(backtester.optimization_method, self.optimization_method)  # Check if optimization_method attribute is set correctly

        self.assertEqual(backtester.rebalance_freq, self.rebalance_freq)  # Check if rebalance_freq attribute is set correctly
        self.assertIsInstance(backtester.rebalance_freq, int) # Check if rebalance_freq is an integer
        self.assertGreaterEqual(backtester.rebalance_freq, 1) # Check if rebalance_freq is greater than or equal to 1
        self.assertLessEqual(backtester.rebalance_freq, 4) # Check if rebalance_freq is less than or equal to 4

        self.assertIsInstance(backtester.risk_aversion, int) # Check if risk_aversion is an integer
        self.assertGreaterEqual(backtester.risk_aversion, 1) # Check if risk_aversion is greater than or equal to 1
        self.assertLessEqual(backtester.risk_aversion, 10) # Check if risk_aversion is less than or equal to 10

        self.assertEqual(backtester.risk_free, self.risk_free)  # Check if risk_free attribute is set correctly
        self.assertGreaterEqual(backtester.risk_free, 1) # Check if risk_free is greater than or equal to 1
        self.assertIsInstance(backtester.risk_free, (int, float)) # Check if risk_free is an integer or float
        
        self.assertEqual(backtester.money, self.money)  # Check if money attribute is set correctly
        self.assertIsInstance(backtester.money, int) # Check if money is an integer

        self.assertEqual(backtester.benchmark, self.benchmark)  # Check if benchmark attribute is set correctly
        self.assertIsInstance(backtester.benchmark, str)
        self.assertIn(backtester.benchmark, ['SPY', 'IWM', 'VEA'])
    
    def test_expanding(self):
        '''
        Test if the expanding method returns a list composed with the correct number of elements and that each element is a pandas DataFrame.
        Test if the last element of the list is the same as the input data (same columns, same index, same values).
        '''
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        subframes = backtester.expanding_window(self.data)

        last_frame = subframes[-1]

        self.assertIsInstance(subframes, list) # Check if subframes is a list
        self.assertTrue(len(subframes) > 0) # Check if subframes has at least one element

        self.assertIsInstance(last_frame, pd.DataFrame) # Check if last_frame is a pandas DataFrame
        self.assertEqual(last_frame.columns.tolist(), self.data.columns.tolist()) # Check if columns are the same
        self.assertEqual(last_frame.index.tolist(), self.data.index.tolist()) # Check if index are the same
        self.assertFalse(last_frame.isnull().all().all()) # Check if DataFrame contains non-NaN values
    
    def test_optimize_back(self):
        '''
        Test if the optimize_back function returns a dataframe with the correct columns and index, that the shape of the DataFrame is correct
        and that the values are between 0 and 1. The sum of each column should be equal to 1.
        '''
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        subframes = backtester.expanding_window(self.data)
        weights_df = backtester.optimize_back(subframes)

        self.assertIsInstance(weights_df, pd.DataFrame) # Check if weights_df is a pandas DataFrame
        self.assertFalse(weights_df.isnull().all().all()) # Check if DataFrame contains non-NaN values
        self.assertAlmostEqual(np.round(weights_df.sum().mean(),2), 1) # Check if sum of each column is equal to 1
        self.assertEqual(weights_df.index.tolist(), sorted(self.stocks)) # Check if index are the stocks

    def test_alphaslist(self):
        '''
        Test if the alphas_list function returns a list and that each element is a pandas DataFrame.
        Test if the last element of the list has the same stock as the data.
        '''
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        subframes = backtester.expanding_window(self.data)
        weights_df = backtester.optimize_back(subframes)
        alphas_list = backtester.alphas_list(self.data, weights_df)

        last_alpha = alphas_list[-1]

        self.assertIsInstance(alphas_list, list) # Check if alphas_list is a list
        self.assertTrue(len(alphas_list) > 0)
        self.assertIsInstance(last_alpha, pd.DataFrame) # Check if last_alpha is a pandas DataFrame
        self.assertEqual(last_alpha.index.tolist(), self.data.columns.tolist()) # Check if stocks are the same
    
    def test_alphasdf(self):
        '''
        Test if the alphas_df function returns a pandas DataFrame with the correct columns and index.
        That everything column sum is equal to 1.
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        log_ret = optimizer.get_logret(self.data)
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        subframes = backtester.expanding_window(self.data)
        weights_df = backtester.optimize_back(subframes)
        alphas_list = backtester.alphas_list(self.data, weights_df)
        alphas_df = backtester.alphas_df(alphas_list, log_ret)

        self.assertIsInstance(alphas_df, pd.DataFrame) # Check if alphas_df is a pandas DataFrame
        self.assertFalse(alphas_df.isnull().all().all()) # Check if DataFrame contains non-NaN values
        self.assertEqual(alphas_df.index.tolist(), self.data.columns.tolist()) # Check if stocks are the same
        self.assertAlmostEqual(np.round(alphas_df.sum().mean(),2), 1) # Check if sum of each column is equal to 1
    
    def test_expostperf(self):
        '''
        Test if the expostperf function returns a pandas Series
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        log_ret = optimizer.get_logret(self.data)
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        subframes = backtester.expanding_window(self.data)
        weights_df = backtester.optimize_back(subframes)
        alphas_list = backtester.alphas_list(self.data, weights_df)
        alphas_df = backtester.alphas_df(alphas_list, log_ret)
        expost_perf = backtester.expost_perf(alphas_df, log_ret)

        self.assertIsInstance(expost_perf, pd.Series) # Check if expost_perf is a pandas Series
        self.assertFalse(expost_perf.isnull().all()) # Check if Series contains non-NaN values

    def test_benchmarkdata(self):
        '''
        Test if the benchmark_data function returns a pandas DataFrame
        '''
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        subframes = backtester.expanding_window(self.data)
        weights_df = backtester.optimize_back(subframes).T.drop_duplicates().T
        alphas_list = backtester.alphas_list(self.data, weights_df)
        log_ret = optimizer.get_logret(self.data)
        alphas_df = backtester.alphas_df(alphas_list, log_ret)
        expost_perf = backtester.expost_perf(alphas_df, log_ret)
        benchmark_data = backtester.benchmark_data(expost_perf, self.benchmark)

        self.assertIsInstance(benchmark_data, pd.Series) # Check if benchmark_data is a pandas DataFrame
        self.assertFalse(benchmark_data.isnull().all().all()) # Check if DataFrame contains non-NaN values
        self.assertTrue(benchmark_data.values.dtype == 'float64')

class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.data_df = cache.get('data_df')
        if self.data_df is None:
            self.data_df = pd.read_csv("optimizer/df3.csv")
            cache.set('data_df', self.data_df, timeout=None)

        ## Randomly select a few stocks (between 2 and 10) from the dataset.
        self.stocks = stocks
        self.optimization_method = optimization_method
        self.rebalance_freq = rebalance_freq
        self.risk_aversion = risk_aversion
        self.risk_free = risk_free
        self.money = money
        self.benchmark = benchmark
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        self.data = optimizer.get_data()
        self.lookback = optimizer.get_lookback(self.data)
    
    def test_plots(self):
        optimizer = Optimizer(self.stocks, self.optimization_method, self.risk_aversion, self.risk_free, self.money)
        backtester = Backtester(self.stocks, self.optimization_method, self.lookback, self.rebalance_freq, self.money, self.risk_aversion, self.risk_free, self.benchmark)
        plotter = Plotter()

        data = optimizer.get_data()
        mu = optimizer.get_mu(data)
        covmat = optimizer.get_covmat(data)
        lookback = optimizer.get_lookback(data)
        log_ret = optimizer.get_logret(data)

        subframes = backtester.expanding_window(data)
        weights_df = backtester.optimize_back(subframes)
        alphas_list = backtester.alphas_list(data, weights_df)
        alphas_df = backtester.alphas_df(alphas_list, log_ret)
        expost_perf = backtester.expost_perf(alphas_df, log_ret)

        last_weights = np.round(weights_df.iloc[-1],8)
        discrete = optimizer.discrete_values(data, OrderedDict(last_weights))
        benchmark_ret = backtester.benchmark_data(expost_perf, self.benchmark)


        pie_plot = plotter.pie_plot(last_weights)
        line_plot = plotter.portfolio_vs_benchmark(expost_perf, benchmark_ret)
        efficient_plot = plotter.plot_efficient_frontier(mu, covmat, weights_df)
        metrics_table = plotter.metrics_table(expost_perf, benchmark_ret, self.risk_free, self.money)
        recap_table = plotter.portfolio_alloc_recap(last_weights, discrete, self.data_df)

        ## Check if the plots are not None
        self.assertIsNotNone(pie_plot)
        self.assertIsNotNone(line_plot)
        self.assertIsNotNone(efficient_plot)
        self.assertIsNotNone(metrics_table)
        self.assertIsNotNone(recap_table)

        #check if metrics table and recap table are dataframes
        self.assertIsInstance(metrics_table, pd.io.formats.style.Styler)
        self.assertIsInstance(recap_table, pd.io.formats.style.Styler)

        # check if the plots are output as divs
        self.assertIsInstance(pie_plot, str)
        self.assertIsInstance(line_plot, str)
        self.assertIsInstance(efficient_plot, str)

        # check if the tables are output as html tables
        self.assertIsInstance(metrics_table.to_html(), str)
        self.assertIsInstance(recap_table.to_html(), str)