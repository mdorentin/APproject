# PortfolioWizard

Capstone Project for the Advanced Programming Class @ HEC Lausanne

Author: Dorentin Morina

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white) ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
## Description of the application
PortfolioWizard is a user-friendly web application built on the Django framework, tailored to empower individuals without financial expertise in optimizing their portfolios. Users can personalize their equity portfolios according to their preferences with a wide range of customization features. From selecting stocks to choosing optimization methods, PortfolioWizard gives you full control. Powered by an algorithm based on Nobel Prize winner Markowitz's Modern Portfolio Theory and its variations, this tool ensures robust portfolio management.

## Functions of the application
- Imports historical data price given user's preference..
- Performs an optimization given the user's parameters (optimization method, risk aversion, risk-free rate, budget, and rebalance frequency.)
- Backtests the strategy and compares the result with the user's preferred benchmark.
- Renders a result page, which contains insightful tables and plots.

## Screenshots
![image](https://github.com/mdorentin/APproject/assets/72168825/055e762a-b7df-4e0e-b80c-dc571acdef0f)

![image](https://github.com/mdorentin/APproject/assets/72168825/aa755355-ab8a-45b7-a8f8-628d67e47397)

![image](https://github.com/mdorentin/APproject/assets/72168825/c95cae3e-73f8-450b-987a-94f1da10ebf7)

![image](https://github.com/mdorentin/APproject/assets/72168825/895aa900-4df5-4bdd-94e6-73db64ad348a)

## Notable Packages

PortfolioWizard uses a number of open-source projects to work properly:

- [yfinance] - To import the historical data price.
- [PyPortfolioOpt] - To perform the optimization.
- [Plotly] - To make some nice plots.

## How To Install It
Clone the project and install the virtual environment with the necessary packages:

```sh
git clone https://github.com/mdorentin/APproject.git
cd APproject
pip install virtualenv
virtualenv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```
## Disclaimer
This web application provides portfolio optimization results for educational purposes only. It does not offer financial advice; users should seek professional advice before making investment decisions. Past performance does not indicate future results, and all investments carry risks.
