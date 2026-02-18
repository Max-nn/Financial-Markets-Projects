import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

# Step 2 - Importing Fama-French Data

data = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)
data_df = data[0].div(100) #data is initially displayed as percentage, we convert it to decimal to avoid wrong calculations
print(data_df.head()) # Inspect data
data_df = data_df.rename(columns={'Mkt-RF': 'Mkt_RF'})

# Step 3 - Importing Security Data

ticker = input('Enter Yahoo Finance ticker:',)
df_asset = yf.download(ticker, start=start, end=end, interval="1mo")
# We only keep close
prices = df_asset['Close']
print(df_asset)
# calculation of monthly returns
monthly_returns = prices.pct_change().dropna()

# Step 4 - Merging Dataframes
# We first need to modify the time index of the yahoo finance dataframe fama = "YYYY-MM" ; yf= "YYYY-MM-DD"

monthly_returns = monthly_returns.to_period('M')
merged_df = pd.merge(data_df, monthly_returns, left_index=True, right_index=True, how='inner').rename(columns={monthly_returns.columns[-1]: 'Asset_return'})
print(merged_df)

# Step 5 - Feature engineering

# Dependant Variable (Y)
merged_df['Excess_Return'] = merged_df['Asset_return'] - merged_df['RF']
Y = merged_df['Excess_Return']
# Independent Variable (X)
X = merged_df[['Mkt_RF', 'SMB', 'HML']]
X = sm.add_constant(X) # otherwise regression line will be forced to pass through the origin

# Step 6 - Econometric Model

model = sm.OLS(Y, X).fit()
model_summary = model.summary()
print(model_summary)

# Step 7 - Quantitative analysis & Attribution

# 7.1 - We first get the intercept term  and analyze it ('const')
alpha = model.params['const']  # This is the monthly alpha, we need to convert it annually
alpha = (1 + alpha) ** 12 - 1   # annual alpha
p_value_alpha = model.pvalues['const']
if p_value_alpha < 0.05:
    print('Alpha is statistically significant')
else: print(f'Alpha is NOT statistically significant: p-value Alpha =  {p_value_alpha}\n')

# 7.2 - We then get the betas term  and analyze it ('Mkt_RF', 'SMB', 'HML')

beta_MKT , beta_SMB, beta_HML = model.params[['Mkt_RF','SMB','HML']]
print(f"Beta Mkt_RF: {beta_MKT}")

# Investment style analysis
# SMB
if beta_SMB > 0.2:
    print(f"Beta SMB = {beta_SMB} => Small Cap Bias detected")
elif beta_SMB < -0.2:
    print(f"Beta SMB = {beta_SMB} => Large Cap Bias detected")
else:
    print(f"Beta SMB = {beta_SMB} => No bias detected")

# HML
if beta_HML > 0.2:
    print(f"Beta SMB = {beta_SMB} => Value Cap Bias detected")
elif beta_HML < -0.2:
    print(f"Beta SMB = {beta_SMB} => Growth Bias detected")
else:
    print(f"Beta SMB = {beta_SMB} => No bias detected")

# 7.3 - Risk Analysis of Active Management

tracking_error = model.resid * np.sqrt(12)
information_ratio = alpha / tracking_error  # alpha is already annualized
unsystematic_risk = 1 - model.rsquared  # Risk that isn't diversified away (specific risk of the security)

# Step 8 - Visualisation
# Graph 1 = Linear Regression of Excess Return vs. Market Excess Return
intercept = alpha

for factor in ['Mkt_RF','SMB','HML']:
    if model.pvalues[factor] < 0.05:
        if factor != 'Mkt_RF':
            x_line = np.linspace(merged_df[factor].min(), merged_df[factor].max(), 100)
            y_line = alpha + (model.params[factor] * x_line)
            plt.scatter(merged_df[factor] * 100, Y * 100, alpha=0.5, color='gray', label='Historical Data')
            plt.plot(x_line * 100, y_line * 100, color='red', lw=2, label=f'Regression Line (B={model.params[factor]:.2f})')
            plt.title(f"Linear Regression with factor {factor}")
            plt.show()
        else:
            x_line = np.linspace(merged_df[factor].min(), merged_df[factor].max(), 100)
            y_line = alpha + (model.params[factor] * x_line)
            plt.scatter(merged_df[factor] * 100, Y * 100, alpha=0.5, color='gray', label='Historical Data')
            plt.plot(x_line * 100, y_line * 100, color='red', lw=2,
                     label=f'Regression Line (B={model.params[factor]:.2f})', linestyle='--')
            plt.title(f"Linear Regression with factor {factor}")
            plt.show()
    else: print(f"Factor {factor} isn't statistically significant\n PLOTTING CANCELED")

# Graph 2 = Beta MKT 12 months rolling window plotting

window = 12
# 1 - Rolling covariance (security, market)
rolling_cov = Y.rolling(window=window).cov(merged_df['Mkt_RF'])

# 2 - Rolling market variance
rolling_var = merged_df['Mkt_RF'].rolling(window=window).var()

# 3 - Rolling Beta
rolling_beta = rolling_cov / rolling_var

# Plotting Rolling Beta
plt.plot(merged_df.index.to_timestamp(), rolling_beta, label=f'Rolling Beta ({window}m)', color='blue')

# Reference line B=1
plt.axhline(1, color='red', linestyle='--', alpha=0.6, label='Market Beta (B=1)')

# Static Beta line
plt.axhline(model.params['Mkt_RF'], color='green', linestyle=':', label='Static Beta')

plt.title("Evolution of Market Risk (Rolling Beta)")
plt.legend()
plt.show()
