from django import forms

class InvestmentForm(forms.Form):
    ### Stock Input
    stocks = forms.CharField(label='Stock tickers', widget=forms.TextInput(attrs={'class': 'stock-input'}))

    ### Optimization Parameters
    optimization_method = forms.ChoiceField(label='Optimization method',
                                            choices=[('equal', 'Equally Weighted'),
                                                     ('max_sr', 'Sharpe Ratio Maximization'),
                                                     ('min_vol', 'Volatility Minimization'),
                                                     ('min_cvar', 'Conditional Value at Risk Minimization'),
                                                     ('max_quad', 'Quadratic Utility Maximization'),
                                                     ('hrp', 'Hierarchical Risk Parity')],
                                            widget=forms.Select(attrs={'class': 'parameter-input'}))
    
    risk_aversion = forms.IntegerField(label='Risk aversion coefficient', min_value=1, max_value=10,
                                        widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    risk_free = forms.FloatField(label='Risk-free rate', min_value=1,
                                  widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    rebalance_freq = forms.ChoiceField(label='Rebalance frequency',
                                       choices=[('1', 'Annually'), ('2', 'Semi-annually'), ('4', 'Quarterly')],
                                        widget=forms.Select(attrs={'class': 'parameter-input'}))
    
    money = forms.IntegerField(label='Initial investment', min_value=1000,
                               widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    
    benchmark = forms.ChoiceField(label='Benchmark',
                                  choices=[('SPY', 'SPDR S&P 500 ETF (SPY)'), ('IWM', 'iShares Russell 2000 ETF (IWM)'), ('VEA', 'Vanguard FTSE Developed Markets ETF (VEA)')],
                                  widget=forms.Select(attrs={'class': 'parameter-input'}))
