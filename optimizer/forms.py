from django import forms

class InvestmentForm(forms.Form):
    ### Stock Input
    stocks = forms.CharField(label='Stock tickers', widget=forms.TextInput(attrs={'class': 'stock-input'}))

    ### Optimization Parameters
    optimization_method = forms.ChoiceField(label='Optimization method',
                                            choices=[('max_sr', 'Sharpe Ratio Maximization'),
                                                     ('min_vol', 'Volatility Minimization'),
                                                     ('min_cvar', 'Conditional Value at Risk Minimization'),
                                                     ('max_quad', 'Quadratic Utility Maximization'),
                                                     ('hrp', 'HRP'),
                                                     ('equal', 'Equally Weighted')],
                                            widget=forms.Select(attrs={'class': 'parameter-input'}))
    
    lookback = forms.IntegerField(label='Lookback period', min_value=1, max_value=10,
                                  widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    risk_aversion = forms.IntegerField(label='Risk aversion coefficient', min_value=1, max_value=10,
                                        widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    risk_free = forms.FloatField(label='Risk-free rate', min_value=1,
                                  widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    rebalance_freq = forms.ChoiceField(label='Rebalance frequency',
                                       choices=[('1', 'Annually'), ('2', 'Semi-annually'), ('4', 'Quarterly')],
                                        widget=forms.Select(attrs={'class': 'parameter-input'}))
    
    money = forms.IntegerField(label='Initial investment', min_value=1,
                               widget=forms.NumberInput(attrs={'class': 'parameter-input'}))
    
    
    benchmark = forms.ChoiceField(label='Benchmark',
                                  choices=[('SPY', 'SPY'), ('SPY', 'SPY'), ('SPY', 'SPY')],
                                  widget=forms.Select(attrs={'class': 'parameter-input'}))
