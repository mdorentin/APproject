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

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Optimizer:
    '''
    Optimizer class for portfolio optimization.
    
    Attributes:
        * stocks (list): A list of stock tickers for the equity portfolio.
        * optimization_method (str): The method chosen for portfolio optimization.
        * lookback (int): Number of years of historical data considered for optimization.
        * risk_aversion (int): User's risk aversion parameter. Only used in max_quad optimization.
        * money (int): Amount of money available for investment. Also the starting portfolio value during the backtest.
    '''
    
    def __init__(self, stocks, optimization_method, lookback, risk_aversion, risk_free, money):
        '''
        Initializes the Optimizer with user-defined parameters.
        '''
        self.stocks = stocks
        self.optimization_method = optimization_method
        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self.risk_free = risk_free
        self.money = money
    
    def get_data(self):
        '''
        Fetches historical stock price data for the specificed stocks.
            - Imported from Yahoo Finance with the yfinance package.
            
        Returns:
            * pandas.DataFrame: Historical stock price data.
        '''
        today = dt.date.today()
        start = today.replace(year=today.year - (self.lookback + 3))
        
        try:
            data = yf.download(self.stocks, start=start, progress=False)['Close']        
            if data.iloc[-1,:].isna().any():
                data = data.iloc[:-1,:]
                
            return data
        
        except Exception as e:
            print(f'An error occured during the data import: {e}')
    
    def get_logret(self, data):
        '''
        Computes the log returns of the provided data.
        
        Returns:
            * pandas.DataFrame: Log returns data.
        '''
        try:
            logret = np.log(data/data.shift(1)).dropna()
            return logret
        except Exception as e:
            print(f'An error occured while computing the log returns: {e}')
            
    def get_covmat(self, data):
        '''
        Computes the covariance matrix of the provided data.
            - Use the CovarianceShrinkage function from the pypfopt package.
            - This method was used, because despite being biased, it has way less estimation error.
            - Ledoit Wolf method was choosed to estimate the optimal shrinkage constant.
            
        Returns:
            * pandas.DataFrame: Covariance matrix.
        '''
        try:
            covmat = risk_models.CovarianceShrinkage(data, log_returns=True).ledoit_wolf()
            return covmat
        except Exception as e:
            print(f'An error occured while computing the covariance matrix: {e}')
            
    
    def get_mu(self, data):
        '''
        Computes the expected returns of the provided date.
            - Use the Exponential Moving Average (EMA) function from the pypfopt package.
            - This method was used, because it is an improvement over the mean historical returns.
            - It gives more weights to recent returns and thus increase the relevance of the estimates.
            
        Returns:
            * pandas.Series: Expected returns.
        '''
        try:
            mu = expected_returns.ema_historical_return(data, log_returns=True)
            return mu
        except Exception as e:
            print(f'An error occured while computing the expected returns: {e}')
    
    def optimize(self, data, covmat, mu):
        '''
        Finds optimal portfolio weights using the specified optimization method.
        
        Returns:
            * OrderedDict: Optimal portfolio weights
        '''
        n_stock = len(data.T)
        ef = EfficientFrontier(mu, covmat)
        
        if self.optimization_method == 'equal':
            weights = OrderedDict(zip(data.columns.values, np.ones(n_stock) / n_stock))
        
        elif self.optimization_method == 'max_sr':
            try:
                weights = ef.max_sharpe(risk_free_rate=self.risk_free)
            except Exception as e:
                print(f'An error occured during the maximum Sharpe Ratio Optimization: {e}')
        
        elif self.optimization_method == 'min_vol':
            try:
                weights = ef.min_volatility()
            except Exception as e:
                print(f'An error occured during the minimium Volatility Optimization: {e}')
        
        elif self.optimization_method == 'max_quad':
            try:
                weights = ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
            except Exception as e:
                print(f'An error occured during the maximum quadratic utility function optimization: {e}')    
        
        elif self.optimization_method == 'min_cvar':
            try:
                ef = EfficientCVaR(mu, covmat)
                weights = ef.min_cvar()
            except Exception as e:
                print(f'An error occured during the minimum CVaR optimization: {e}')  
        
        elif self.optimization_method == 'hrp':
            try:
                returns = expected_returns.returns_from_prices(data)
                hrp = HRPOpt(returns)
                hrp.optimize()
                weights = hrp.clean_weights()
            except Exception as e:
                print(f'An error occured during the Hierarchial Risk Parity Optimization: {e}')
        
        else:
            print('Please insert a valid optimization method!')
            
        return weights   
        
    def discrete_values(self, data, weights):
        '''
        Computes the number of stocks units to buy for each stock in the portfolio.
          
        Returns:
            * pandas.DataFrame: DataFrame showing the number of units to buy for each stock.
        '''
        try:
            latest_prices = data.iloc[-1]
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=self.money)
            discrete = pd.DataFrame.from_dict(da.lp_portfolio()[0], orient='index', columns=[latest_prices.name])
            return discrete
        except Exception as e:
            print(f'An error occured while computation the discrete number of stocks units to buy for each stock: {e}')
        

class Backtester:
    '''
    Backtester class for portfolio optimization.
    
    Attributes:
        * optimization_method (str): The method chosen for portfolio optimization.
        * lookback (int): Number of years of historical data considered for optimization.
        * rebalance_freq (int): Number of times that we rebalance our portfolio per year. (1=annually, 2=semi-annually, ...)
        * money (int): Amount of money available for investment. Also the starting portfolio value during the backtest.
        * benchmark (str): Benchmark ETF or Indices to compare with the performance of our portfolio.
    '''
    def __init__(self, optimization_method, lookback, rebalance_freq, money, risk_aversion, risk_free, benchmark):
        '''
        Initializes the Backtester and the Optimizer with user-defined parameters.
        '''
        self.optimization_method = optimization_method
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.money = money
        self.risk_aversion = risk_aversion
        self.risk_free = risk_free
        self.benchmark = benchmark
    
    def expanding_window(self, data):
        '''
        Splits the dataframe into multiples subdataframes according to the lookback period and the rebalance_freq.
        
        Returns:
            * List: List of all the subdataframes
        '''
        df_list = []
        total_rebalances = int(self.lookback * self.rebalance_freq + 1)
        begin = data.index.min()
        
        try:
            for i in range(total_rebalances):
                end_idx = int((252 * 3) + (252/self.rebalance_freq) * i)
                end_date = data[:end_idx].index.max()
                df = data[data.index.isin(pd.date_range(begin, end_date))]
                df_list.append(df)
        
            return df_list
        
        except Exception as e:
            print(f'An error occured while dividing the dataframe into datasubframes.')
    
    def optimize_back(self, df_list):
        '''
        Finds the optimal weights for each of the subdataframes. Useful to backtest our strategy.
        
        Returns:
            * pandas.DataFrame: Dataframe with the % weights every time we rebalanced our portfolio.
        '''
        weights_df = []
        stocksticker = df_list[0].columns
        try:
            optimizer = Optimizer(stocks=stocksticker, optimization_method=self.optimization_method, lookback=self.lookback,
                                  risk_aversion=self.risk_aversion, risk_free=self.risk_free, money=self.money)
        except Exception as e:
            print(f'An error occured while initializing the Optimizer class: {e}')
            
        try:   
            for i in range(len(df_list)):
                data = df_list[i]
                covmat = optimizer.get_covmat(data)
                mu = optimizer.get_mu(data)
                weights = optimizer.optimize(data, covmat, mu)
                weights = pd.DataFrame(weights.items(), columns=['Ticker', data.index.max()]).set_index('Ticker')[data.index.max()]

                weights_df.append(weights)

            return pd.concat(weights_df, axis=1).fillna(0)
        except Exception as e:
            print(f'An error occured while optimizing all the datasubframes: {e}')
    
    def alphas_list(self, data, weights_df):
        '''
        Create all the alphas subdataframes to later compute all the alphas.
        
        Returns:
            * List: List of all the subdataframes
        '''
        alphas_sub = []
        try:
            for i in range(len(weights_df.T)-1):
                start=weights_df.iloc[:,i].name
                end= weights_df.iloc[:,i+1].name - dt.timedelta(days=1)
                dates = data.index[data.index.isin(pd.date_range(start,end))]
                sub_df = pd.DataFrame(index=weights_df.index, columns=dates)
                sub_df.iloc[:,0] = weights_df.iloc[:,i]

                alphas_sub.append(sub_df)
                
            return alphas_sub
        except Exception as e:
            print(f'An error occured while creating the alphas datasubframes: {e}')
    
    def alphas_df(self, alphas_sub, log_ret):
        '''
        Compute all the alphas and create the final alphas dataframe.
        
        Returns:
            * pandas.DataFrame: Dataframe with the alphas from the beginning until the end.
        '''
        try:
            alphas_df = pd.DataFrame()
            
            for i in range(len(alphas_sub)):
                sub = alphas_sub[i].copy()

                begin = sub.iloc[:, 1].name
                end = sub.iloc[:, -1].name
                logrets = log_ret[log_ret.index.isin(pd.date_range(begin, end))].T

                for date in range(1, len(sub.columns)):
                    for stock in range(len(sub)):
                        stock_ret = logrets.iloc[stock, date-1]
                        pf_ret = sub.iloc[:, date-1] @ logrets.iloc[:, date-1]
                        previous_alpha = sub.iloc[stock, date-1]
                        sub.iloc[stock, date] = previous_alpha * ((1 + stock_ret) / (1 + pf_ret))

                alphas_df = pd.concat([alphas_df, sub], axis=1)
            return alphas_df
        except Exception as e:
            print(f'An error occured while computing the alphas: {e}')
    
    def expost_perf(self, alphas_df, log_ret):
        '''
        Compute the ex-post performance of our portfolio.
        
        Returns:
            * pandas.Series: Series with the daily performance logreturn of our portfolio.
        '''
        start = alphas_df.columns[0]
        end = alphas_df.columns[-1]

        logret = log_ret[log_ret.index.isin(pd.date_range(start, end))].T
        try:
            daily_perf = logret.multiply(alphas_df).sum(axis=0).astype('float64')
            return daily_perf
        except Exception as e:
            print(f'An error occured while computing the ex-post performances: {e}')
    
    def benchmark_data(self, data, benchmark):
        '''
        Fetches the benchmark historical data price and computes the log returns.
        
        Returns:
            * pandas.Series: Series with the daily performance logreturn of the benchmark.
        '''
        try:
            start = data.index.min()
            df = yf.download(benchmark, start, progress=False)['Close']

        except Exception as e:
            print(f'An error occured during the benchmark data import: {e}')
        
        try:
            logret = np.log(df/df.shift(1)).dropna()
            return logret
        except Exception as e:
            print(f'An error occured while computing the benchmark log returns: {e}')
            

class Plotter():
    '''
    Plotter class for portfolio optimization.
    
    Attributes: None
    '''
    def __init__(self):
        pass
        
    def pie_plot(self, weights_df):
        '''
        Plot the pie chart showing the weights of our portfolio.
        
        Returns:
            * plotly.Figure: Pie chart.
        '''
        last_weights = weights_df.iloc[:,-1]
        fig = px.pie(last_weights.values, values=0, names=last_weights.index)

        fig.update_traces(textposition='inside', 
                              textinfo='label',
                              hoverinfo='percent',
                              hovertemplate='%{percent:.2%} (%{value})',  # Set hoverinfo as percentage with 2 decimals
                              textfont=dict(color='white'))  # Set text color to white

        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig.update_layout(font=dict(color='white'))
        fig.update_layout(title='Pie Chart of Portfolio Weights')

        return plot(fig, output_type='div')
        #return fig
    
    def portfolio_vs_benchmark(self, expost_perf, benchmark_ret):
        '''
        Plot the cumulative returns of our portfolio and of the benchmark.
        
        Returns:
            * plotly.Figure: Line chart.
        '''
        pf_cumprod = (1+expost_perf).cumprod() - 1
        bench_cumprod = (1+benchmark_ret).cumprod() - 1
        cum_ret = pd.concat([pf_cumprod, bench_cumprod], axis=1).dropna()
        cum_ret.columns = ['Portfolio', 'Benchmark']
        
        fig = px.line(cum_ret, labels={'index': 'Index', 'value': 'Cumulative Return', 'variable': ''})
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        # Set label color to white
        fig.update_layout(xaxis=dict(title=dict(text='Index', font=dict(color='white'))))
        fig.update_layout(yaxis=dict(title=dict(text='Cumulative Return', font=dict(color='white'))))

        # Set text color to white
        fig.update_layout(legend=dict(font=dict(color='white')))

        # Set x and y numbers color to white
        fig.update_layout(xaxis=dict(tickfont=dict(color='white')))
        fig.update_layout(yaxis=dict(tickfont=dict(color='white')))

        # Make the grid less visible
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)'))
        fig.update_layout(yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)'))
        
        # Add title
        fig.update_layout(title='Portfolio vs. Benchmark Cumulative Returns', font=dict(color='white'))

        return plot(fig, output_type='div')
        #return fig
    
    def plot_efficient_frontier(self, mu, covmat, weights_df):
        '''
        Plot the efficient frontier
        
        Returns:
            * plotly.Figure: Efficient frontier plot
        '''
        ########## Calculated returns and std for our optimized portfolio ############
        ret = (weights_df.iloc[:,-1] @ mu)
        std = (np.sqrt(weights_df.iloc[:,-1] @ covmat @ weights_df.iloc[:,-1].T))

        ############## Generate random portfolios and compute its metrics ############
        num_portfolios = 1000
        weights_random = np.random.dirichlet(np.ones(len(mu)), num_portfolios)
        returns_random = np.dot(weights_random, mu)

        std_random = np.zeros(num_portfolios)
        for i in range(num_portfolios):
            std_random[i] = np.sqrt(np.dot(weights_random[i], np.dot(covmat, weights_random[i])))


        ######### Plot the efficient frontier #########
        ef = EfficientFrontier(mu, covmat)

        plotting.plot_efficient_frontier(ef, show_assets=False, plot_assets=False, plot_star=False)

        ######### Create an empty figure ########
        fig = go.Figure()

        for trace in plt.gca().get_lines():
            fig.add_trace(go.Scatter(x=trace.get_xdata(), y=trace.get_ydata(), mode='lines', name='Efficient Frontier'))
            plt.close()

        fig.add_trace(go.Scatter(x=[std], y=[ret], mode='markers', marker=dict(color='red', size=10), name="Optimized Portfolio"))

        ######### Add random portfolios trace ############
        fig.add_trace(go.Scatter(
            x=std_random.tolist(),
            y=returns_random.tolist(),
            mode='markers',
            marker=dict(color=returns_random, colorscale='plasma', size=5),
            name='Random Portfolios',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(xaxis_title='Standard Deviation', yaxis_title='Expected Return', font=dict(color='white'), legend=dict(x=0, y=-0.5))
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        # Set text color to white
        fig.update_layout(legend=dict(font=dict(color='white')))

        # Set x and y numbers color to white
        fig.update_layout(xaxis=dict(tickfont=dict(color='white')))

        fig.update_layout(yaxis=dict(tickfont=dict(color='white')))

        # Make the grid less visible
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)'))
        fig.update_layout(yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)'))
        
        # Add title
        fig.update_layout(title='Efficient Frontier Plot')

        return plot(fig, output_type='div')
    
    def portfolio_metrics(self, expost_perf, risk_free, money):
        '''
        Creates the table that will show the portfolio metrics to the user.
        
        Returns:
            * pandas.DataFrame: Portfolio table metrics
        '''

        #EXPECTED ANNUAL RETURN
        expected_daily_return = expost_perf.mean()
        expected_annual_return = expected_daily_return * 252

        #ANNUAL VOLATILITY
        daily_volatility = np.sqrt(((expost_perf - expected_daily_return) ** 2).mean())
        annual_volatility = np.sqrt(252) * daily_volatility

        #SHARPE RATIO
        sharpe_ratio = (expected_annual_return - risk_free)/ annual_volatility

        #MAXIMUM DRAWDOWN
        cumprod_ret = (1 + expost_perf).cumprod()
        peaks = cumprod_ret.cummax()
        drawdowns = ((cumprod_ret - peaks) / peaks).min()

        #IF INVESTED {LOOKBACK} YEARS AGO
        end_money = np.round(money * cumprod_ret.iloc[-1],0)

        ## Reformatting to make it look better
        expected_annual_return = format(expected_annual_return, ".2%")
        annual_volatility = format(annual_volatility, ".2%")
        drawdowns = format(drawdowns, ".2%")
        sharpe_ratio = format(sharpe_ratio, ".2f")
        end_money = '${:,.0f}'.format(end_money)

        table_metrics = pd.DataFrame([expected_annual_return,annual_volatility,sharpe_ratio,drawdowns,end_money],
                                     index=['Expected Annual Return','Annual Volatility','Sharpe Ratio','Maximum Drawdown',
                                            f'Valuation if invested in {expost_perf.index.min().year}'],
                                     columns=['Optimized Portfolio'])


        table_metrics

        return table_metrics
    
    def benchmark_metrics(self, expost_perf, benchmark_ret, risk_free, money):
        '''
        Creates the table that will show the benchmark metrics to the user.
        
        Returns:
            * pandas.DataFrame: Benchmark table metrics
        '''

        expected_daily_return = benchmark_ret.mean()
        expected_annual_return = expected_daily_return * 252

        #ANNUAL VOLATILITY
        daily_volatility = np.sqrt(((benchmark_ret - expected_daily_return) ** 2).mean())
        annual_volatility = np.sqrt(252) * daily_volatility

        #SHARPE RATIO
        sharpe_ratio = (expected_annual_return - risk_free)/ annual_volatility

        #MAXIMUM DRAWDOWN
        cumprod_ret = (1 + benchmark_ret).cumprod()
        peaks = cumprod_ret.cummax()
        drawdowns = ((cumprod_ret - peaks) / peaks).min()

        #IF INVESTED {LOOKBACK} YEARS AGO
        end_money = np.round(money * cumprod_ret.iloc[-1],0)

        ## Reformatting to make it look better
        expected_annual_return = format(expected_annual_return, ".2%")
        annual_volatility = format(annual_volatility, ".2%")
        drawdowns = format(drawdowns, ".2%")
        sharpe_ratio = format(sharpe_ratio, ".2f")
        end_money = '${:,.0f}'.format(end_money)

        table_metrics = pd.DataFrame([expected_annual_return,annual_volatility,sharpe_ratio,drawdowns,end_money],
                             index=['Expected Annual Return','Annual Volatility','Sharpe Ratio','Maximum Drawdown',
                                    f'Valuation if invested in {expost_perf.index.min().year}'],
                             columns=['Benchmark'])

        return table_metrics
    
    def portfolio_alloc_recap(self, weights_df, discrete):
        '''
        Creates the table that will show the company's ticker, name, sector and finally weights in % and also as a quantity of stock to purchase.
        
        Returns:
            * pandas.DataFrame: Portfolio recap
        '''
        df = pd.read_csv("optimizer/data3.csv", index_col=0)

        recap_table = pd.DataFrame(index=weights_df.index, columns=['Company Name', 'Weights', 'Shares'])

        # Assign values to columns using loc accessor
        recap_table.loc[:, 'Company Name'] = df.loc[recap_table.index].iloc[:, 0].values
        recap_table.loc[:, 'Weights'] = np.round(weights_df.iloc[:, -1].values * 100, 2)
        recap_table.loc[:, 'Shares'] = discrete

        recap_table = recap_table.sort_values(by='Weights', ascending=False)

        recap_table['Weights'] = recap_table['Weights'].astype(str) + '%'

        if recap_table.Shares.isna().any():
            recap_table['Shares'] = recap_table['Shares'].fillna(0)
        

        return recap_table