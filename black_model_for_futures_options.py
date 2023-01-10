from nsepython import *
import pandas as pd
from scipy.stats import norm
from scipy import optimize
from math import sqrt, log
import numpy as np
import time
import os
import glob
import warnings
from jugaad_data.nse import index_csv, index_df
from datetime import date

warnings.simplefilter("ignore")

# Define timestamp so that an accurate timestamp is used
timestamp = pd.Timestamp.now().round('min')
folder_path = '/NIFTY/intraday_data/'

# Put
df_pe = pd.DataFrame(nse_optionchain_scrapper('NIFTY')['records']['data'])['PE']
df_pe = df_pe.dropna()
df_pe = df_pe.reset_index()
df_pe = df_pe['PE'].apply(pd.Series)
df_pe = df_pe.rename(columns={
    'identifier': 'put_identifier', 
    'strikePrice': 'strike',
    'expiryDate': 'expiry',
    'underlying': 'symbol', 
    'openInterest': 'put_open_interest',
    'changeinOpenInterest': 'put_change_in_oi',
    'pchangeinOpenInterest': 'put_p_change_in_oi', 
    'impliedVolatility': 'put_nse_implied_volatility',
    'totalTradedVolume': 'put_total_traded_volume', 
    'lastPrice': 'put_price',
    'change': 'put_price_change', 
    'pChange': 'put_p_price_change',
    'totalBuyQuantity': 'put_total_buy_qty',
    'totalSellQuantity': 'put_total_sell_qty',
    'bidQty': 'put_bid_qty',
    'bidprice': 'put_bid_price',
    'askQty': 'put_ask_qty',
    'askPrice': 'put_ask_price',
    'underlyingValue': 'put_spot_price'
})
# df_pe['close_ul'] = nse_quote_ltp('NIFTY',"latest","Fut")
df_pe['expiry'] = df_pe['expiry'] + ' ' + '15:30:00'
df_pe['expiry'] = pd.to_datetime(df_pe['expiry'])
df_pe['strike'] = df_pe['strike'].astype(float)
# df_pe_raw = df_pe
# df_pe = df_pe[df_pe['close_ul'] - df_pe['strike'] >= 0]
# df_pe['moneyness'] = 1.0 * np.log(df_pe['close_ul'] / df_pe['strike'])
# df_pe['implied_volatility'] = ''
df_pe['put_option_type'] = 'PE'
df_pe['timestamp'] = timestamp
df_pe['put_days_to_expiry'] = 1.00 * (df_pe['expiry'] - df_pe['timestamp']) / np.timedelta64(1,'h') / 24.0
df_pe = df_pe[[
    'symbol', 'timestamp', 'strike', 'expiry', 'put_option_type','put_days_to_expiry', 
    'put_identifier', 'put_open_interest', 'put_change_in_oi', 'put_p_change_in_oi', 
    'put_total_traded_volume', 'put_nse_implied_volatility', 'put_price', 
    'put_price_change', 'put_p_price_change', 'put_total_buy_qty', 'put_total_sell_qty',
    'put_bid_qty', 'put_bid_price', 'put_ask_qty', 'put_ask_price',
    'put_spot_price'
]]

# Call
df_ce = pd.DataFrame(nse_optionchain_scrapper('NIFTY')['records']['data'])['CE']
df_ce = df_ce.dropna()
df_ce = df_ce.reset_index()
df_ce = df_ce['CE'].apply(pd.Series)
df_ce = df_ce.rename(columns={
    'identifier': 'call_identifier', 
    'strikePrice': 'strike',
    'expiryDate': 'expiry',
    'underlying': 'symbol', 
    'openInterest': 'call_open_interest',
    'changeinOpenInterest': 'call_change_in_oi',
    'pchangeinOpenInterest': 'call_p_change_in_oi', 
    'impliedVolatility': 'call_nse_implied_volatility',
    'totalTradedVolume': 'call_total_traded_volume', 
    'lastPrice': 'call_price',
    'change': 'call_price_change', 
    'pChange': 'call_p_price_change',
    'totalBuyQuantity': 'call_total_buy_qty',
    'totalSellQuantity': 'call_total_sell_qty',
    'bidQty': 'call_bid_qty',
    'bidprice': 'call_bid_price',
    'askQty': 'call_ask_qty',
    'askPrice': 'call_ask_price',
    'underlyingValue': 'call_spot_price'
})
# df_ce['close_ul'] = nse_quote_ltp('NIFTY',"latest","Fut")
df_ce['expiry'] = df_ce['expiry'] + ' ' + '15:30:00'
df_ce['expiry'] = pd.to_datetime(df_ce['expiry'])
df_ce['strike'] = df_ce['strike'].astype(float)
# df_ce_raw = df_ce
# df_ce = df_ce[df_ce['close_ul'] - df_ce['strike'] >= 0]
# df_ce['moneyness'] = 1.0 * np.log(df_ce['close_ul'] / df_ce['strike'])
# df_ce['implied_volatility'] = ''
df_ce['call_option_type'] = 'CE'
df_ce['timestamp'] = timestamp
df_ce['call_days_to_expiry'] = 1.00 * (df_ce['expiry'] - df_ce['timestamp']) / np.timedelta64(1,'h') / 24.0
df_ce = df_ce[[
    'symbol', 'timestamp', 'strike', 'expiry', 'call_option_type','call_days_to_expiry', 
    'call_identifier', 'call_open_interest', 'call_change_in_oi', 'call_p_change_in_oi', 
    'call_total_traded_volume', 'call_nse_implied_volatility', 'call_price', 
    'call_price_change', 'call_p_price_change', 'call_total_buy_qty', 'call_total_sell_qty',
    'call_bid_qty', 'call_bid_price', 'call_ask_qty', 'call_ask_price',
    'call_spot_price'
]]

# MErge Put and Call
df = pd.merge(left=df_pe, right=df_ce, on=['symbol', 'timestamp', 'expiry', 'strike'], how='inner')

# Calculate ATM Forward
df['timestamp_expiry'] = df['timestamp'].astype(str) + '_' + df['expiry'].astype(str)
atm_forward_temp = df[
    (df['put_price'] != 0)
    & (df['call_price'] != 0)
][['timestamp', 'expiry', 'timestamp_expiry', 'strike', 'put_price', 'call_price']]
atm_forward_temp['forward'] = atm_forward_temp['call_price'] - atm_forward_temp['put_price'] + atm_forward_temp['strike']
atm_forward_temp['f_minus_k_abs'] = (atm_forward_temp['forward'] - atm_forward_temp['strike']).abs()
temp = atm_forward_temp.groupby(['timestamp_expiry'], sort=False)['f_minus_k_abs'].min()
temp = pd.DataFrame(temp)
temp = temp.reset_index()
atm_forward_temp_02  = pd.merge(left=temp, right=atm_forward_temp, on=['timestamp_expiry', 'f_minus_k_abs'], how='inner')
atm_forward_temp_02 = atm_forward_temp_02[['timestamp_expiry', 'forward']]
atm_forward_temp_02 = atm_forward_temp_02.drop_duplicates('timestamp_expiry')
atm_forward_temp_02 = atm_forward_temp_02.reset_index()
atm_forward_temp_02 = atm_forward_temp_02.drop('index', 1)
df = df[
    (df['put_price'] != 0)
    & (df['call_price'] != 0)
].reset_index().drop('index', 1)
df = df.merge(atm_forward_temp_02, on=['timestamp_expiry'], how='left')
df = df.drop('timestamp_expiry', 1)

# Calculate moneyness, etc.
df['moneyness'] = round(1.000000 * np.log (df['forward'] / df['strike']), 6)
df['forward'] = pd.to_numeric(df['forward'], errors='coerce')
df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

# Finding out the strikes which are OTM; IV and greeks will be computed only for the OTM strikes
df['f_minus_k'] = df['forward'] - df['strike']
df['otm_option_type'] = np.where(df['f_minus_k'] <= 0, 'CE', 'PE')

df['close_option'] = np.where(df['otm_option_type'] == 'PE', df['put_price'], df['call_price'])
df['close_option'] = pd.to_numeric(df['close_option'], errors='coerce')

# Calculate 'days_to_expiry'
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['expiry'] = pd.to_datetime(df['expiry'])
df['days_to_expiry'] = 1.00 * (df['expiry'] - df['timestamp']) / np.timedelta64(1,'h') / 24.0

# Calculate 'risk-free rate'
df['rf_rate'] = (np.log(df['forward'] / df['call_spot_price'])) / ((df['days_to_expiry'] + 0.25) / 365.0)

# Calculate IV
N_prime = norm.pdf
N = norm.cdf    
def find_sigma(sigma, v, F, K, T, r, option_type):
    d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        diff = (F * N(d1) -  N(d2) * K) * np.exp(-r * T) - v
    else:
        diff = (- F * N(-d1) +  N(-d2) * K) * np.exp(-r * T) - v
    return diff   
def run_brentq(i):
    try:
        return optimize.brentq(
            find_sigma, 0.01, 100,
            args=(
                i.close_option, 
                i.forward,
                i.strike, 
                i.days_to_expiry / 365.0,
                i.rf_rate,
                i.otm_option_type
            ),
            xtol=1.0e-4
        )
    except:
        return 0
df['implied_volatility'] = df.apply(run_brentq, axis=1)

# Calculate Greeks
df['d1'] = (np.log(df['forward'] / df['strike']) + (0.5 * df['implied_volatility'] ** 2 ) * (df['days_to_expiry'] / 365.0)) / (df['implied_volatility'] * np.sqrt((df['days_to_expiry'] / 365.0)))
df['d2'] = df['d1'] - df['implied_volatility'] * np.sqrt((df['days_to_expiry'] / 365.0))
df['d1'] = pd.to_numeric(df['d1'], errors='coerce')
df['d2'] = pd.to_numeric(df['d2'], errors='coerce')

# Substituting simple variables for long dataframe column names to use for Greeks calculation
r = df['rf_rate']
t = (df['days_to_expiry']) / 365.0
d1 = df['d1']
d2 = df['d2']
F = df['forward']
σ = df['implied_volatility']
K = df['strike']

c1 = (df['days_to_expiry']) == 0
c2 = (((df['days_to_expiry']) != 0) & (df['otm_option_type'] == 'CE'))
c3 = (((df['days_to_expiry']) != 0) & (df['otm_option_type'] == 'PE'))
c4 = (df['days_to_expiry']) != 0
c5 = (((df['days_to_expiry']) == 0) & (df['otm_option_type'] == 'CE') & (df['f_minus_k'] > 0))
c6 = (((df['days_to_expiry']) == 0) & (df['otm_option_type'] == 'PE') & (df['f_minus_k'] < 0))

delta = [0, np.exp(-r*t)*N(d1), np.exp(-r*t)*(-N(-d1)), 1, 1]
gamma = [0, round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10)]
theta = [0, round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4)]
vega = [0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4)]
rho = [0, t*K*np.exp(-r*t)*N(d2), -t*K*np.exp(-r*t)*N(-d2)]
vanna = [0, round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10)]
charm = [0, round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4), round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4)]
volga = [0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4)]
default = 0

df['delta'] = np.select([c1, c2, c3, c5, c6], delta, default=default)
df['gamma'] = np.select([c1, c4], gamma, default=default)
df['theta'] = np.select([c1, c4], theta, default=default)
df['vega'] = np.select([c1, c4], vega, default=default)
df['rho'] = np.select([c1, c2, c3], rho, default=default)
df['vanna'] = np.select([c1, c4], vanna, default=default)
df['charm'] = np.select([c1, c2, c3], charm, default=default)
df['volga'] = np.select([c1, c4], volga, default=default)

# Save file as CSV
filename = time.strftime("%b%d_%Y_%H%M").lower()
df.to_csv(folder_path+filename+'.csv', index=False)

print(filename)