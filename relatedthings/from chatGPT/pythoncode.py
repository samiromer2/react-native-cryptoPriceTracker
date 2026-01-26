def sample_variance(x):
    n = len(x)
    mean_x = sum(x) / n
    return sum((xi - mean_x) ** 2 for xi in x) / (n - 1)


def population_variance(x):
    n = len(x)
    mean_x = sum(x) / n
    return sum((xi - mean_x) ** 2 for xi in x) / n


data = [2, 4, 6, 8]

print("Sample variance:", sample_variance(data))
print("Population variance:", population_variance(data))



#---------------++++++
def conditional_variance(x, I):
    """
    Computes Var(x | I) = E[(x - E(x|I))^2].
    x : list or array of numeric values
    I : list or array of booleans (same length) indicating the condition
    """
    # Select the subset of x where condition I is True
    x_cond = [xi for xi, cond in zip(x, I) if cond]

    if len(x_cond) == 0:
        raise ValueError("No data satisfies the condition I")

    # Conditional mean E(x | I)
    mean_cond = sum(x_cond) / len(x_cond)

    # Conditional variance E[(x - E(x|I))^2]
    var_cond = sum((xi - mean_cond) ** 2 for xi in x_cond) / len(x_cond)

    return var_cond

x = [2, 4, 6, 8, 10]
I = [True, False, True, False, True]  # Condition

print("Conditional variance:", conditional_variance(x, I))

#---------------++++++
def conditional_variance_hybrid_mean(x, I, alpha=0.85):
    """
    Computes conditional variance using a hybrid mean:
    μ = α * E(x | I) + (1 - α) * E(x)
    
    alpha: weighting for conditional mean (0 to 1)
    """
    # Convert to conditional subset
    x_cond = [xi for xi, cond in zip(x, I) if cond]
    if len(x_cond) == 0:
        raise ValueError("No data satisfies the condition I")

    # Conditional mean E(x | I)
    mean_cond = sum(x_cond) / len(x_cond)

    # Unconditional mean E(x)
    mean_uncond = sum(x) / len(x)

    # Hybrid mean
    hybrid_mean = alpha * mean_cond + (1 - alpha) * mean_uncond

    # Conditional variance using hybrid mean
    var_cond_hybrid = sum((xi - hybrid_mean) ** 2 for xi in x_cond) / len(x_cond)

    return var_cond_hybrid

x = [20000, 20500, 21000, 22000, 18000, 19500]  # Bitcoin prices
I = [True, True, False, False, True, False]     # e.g., "during high volume hours"

variance = conditional_variance_hybrid_mean(x, I, alpha=0.9)
print("Conditional Variance (Hybrid Mean):", variance)



#---------------++++++

def conditional_variance_time(epsilon_t, I_t):
    """
    Computes Var(e_t | I_t) = E[(e_t - E(e_t | I_t))^2]
    where epsilon_t is the error/residual at time t.
    
    I_t is the information set: past prices/returns, volume, indicators, etc.
    """
    # Expected value given information set
    mean_cond = sum(I_t) / len(I_t)  # simple example: conditional mean
    
    # Variance given information set
    sigma_t2 = (epsilon_t - mean_cond) ** 2
    return sigma_t2

def rolling_conditional_variance(errors, window=20):
    """
    Computes time-varying conditional variance sigma_t^2
    using rolling window (like simple ARCH model).
    """
    sigma_t2 = []

    for t in range(len(errors)):
        if t < window:
            sigma_t2.append(None)  # not enough data yet
        else:
            window_data = errors[t-window:t]
            var_t = sum((e - sum(window_data)/len(window_data))**2 for e in window_data) / len(window_data)
            sigma_t2.append(var_t)

    return sigma_t2

def garch_11(errors, omega=0.1, alpha=0.1, beta=0.8):
    """
    Computes conditional variance sigma_t^2 for a GARCH(1,1) model.
    errors: time series of residual returns (e_t)
    """
    T = len(errors)
    sigma2 = [0] * T

    # initialize with unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, T):
        sigma2[t] = omega + alpha * errors[t-1]**2 + beta * sigma2[t-1]

    return sigma2
#---------------++++++

def arch_1(returns, a0=0.1, a1=0.8):
    """
    ARCH(1) model:
    sigma_{t+1}^2 = a0 + a1 * r_t^2
    
    returns: list/array of r_t values (e.g., Bitcoin log returns)
    """
    T = len(returns)
    sigma2 = [0] * (T + 1)

    # initialize variance with unconditional variance of ARCH(1)
    sigma2[0] = a0 / (1 - a1)

    for t in range(T):
        sigma2[t+1] = a0 + a1 * (returns[t] ** 2)

    return sigma2

btc_returns = [0.02, -0.015, 0.01, -0.03, 0.025]

sigma2 = arch_1(btc_returns, a0=0.05, a1=0.7)

print("Conditional variances:", sigma2)


#---------------++++++

#pip install yfinance pandas numpy matplotlib

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. DOWNLOAD BITCOIN HISTORICAL PRICE DATA
# ---------------------------------------------------
print("Downloading Bitcoin price data...")

btc = yf.download("BTC-USD", start="2015-01-01", end="2025-01-01")
prices = btc["Close"].dropna()

# ---------------------------------------------------
# 2. COMPUTE LOG RETURNS
# ---------------------------------------------------
returns = np.log(prices / prices.shift(1)).dropna()

# ---------------------------------------------------
# 3. GARCH(1,1) MODEL
# ---------------------------------------------------
def garch_11(returns, omega=0.000001, alpha=0.1, beta=0.85):
    """
    GARCH(1,1):
    σ_t^2 = ω + α * r_{t-1}^2 + β * σ_{t-1}^2
    """
    T = len(returns)
    sigma2 = np.zeros(T)

    # unconditional variance initialization
    sigma2[0] = np.var(returns)

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]

    return sigma2

# Run model
sigma2 = garch_11(returns)

# Convert to DataFrame
garch_df = pd.DataFrame({
    "returns": returns,
    "variance": sigma2,
    "volatility": np.sqrt(sigma2)
}, index=returns.index)

# ---------------------------------------------------
# 4. PLOT VOLATILITY OVER TIME
# ---------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(garch_df.index, garch_df["volatility"], label="GARCH(1,1) Volatility", linewidth=1.5)
plt.title("Bitcoin GARCH(1,1) Estimated Volatility (σ_t)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.legend()
plt.show()


#btc = yf.download("BTC-USD")

#---------------++++++

#pip install yfinance pandas numpy matplotlib scipy


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------
# 1. Download BTC price data
# ----------------------------
btc = yf.download("BTC-USD", start="2015-01-01", end=None, progress=False)
prices = btc["Close"].dropna()
returns = np.log(prices / prices.shift(1)).dropna()  # daily log returns
returns = returns * 100  # scale returns to percentage (optional, helps numerics)

# ----------------------------
# 2. GARCH(1,1) Log-Likelihood
# ----------------------------
def garch11_sigma2(params, rets):
    """
    Compute sigma^2 series for GARCH(1,1) with parameters params = [omega, alpha, beta]
    """
    omega, alpha, beta = params
    T = len(rets)
    sigma2 = np.zeros(T)
    # initialize sigma2[0] with unconditional variance if stationarity holds, else var(rets)
    if alpha + beta < 1:
        sigma2[0] = omega / (1 - alpha - beta)
    else:
        sigma2[0] = np.var(rets)
    for t in range(1, T):
        sigma2[t] = omega + alpha * (rets[t-1] ** 2) + beta * sigma2[t-1]
    return sigma2

def neg_loglik(params, rets):
    omega, alpha, beta = params
    # enforce positivity to avoid invalid sims
    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e12
    sigma2 = garch11_sigma2(params, rets)
    # avoid zero or negative sigma2
    if np.any(sigma2 <= 0):
        return 1e12
    # Gaussian log-likelihood (sum over t)
    ll = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (rets ** 2) / sigma2)
    return np.sum(ll)

# ----------------------------
# 3. Fit GARCH(1,1) by MLE
# ----------------------------
# initial guesses
init = np.array([1e-6, 0.05, 0.90])

# constraints: omega>0, alpha>=0, beta>=0, alpha+beta < 1
bounds = [(1e-12, None), (0.0, 0.9999), (0.0, 0.9999)]
cons = ({'type': 'ineq', 'fun': lambda x: 0.9999 - (x[1] + x[2])})  # alpha + beta < 0.9999

res = minimize(neg_loglik, x0=init, args=(returns.values,), bounds=bounds, constraints=cons, method='SLSQP',
               options={'maxiter': 2000, 'ftol': 1e-8})

if not res.success:
    print("Warning: optimization did not converge:", res.message)

omega_hat, alpha_hat, beta_hat = res.x
print("Fitted GARCH(1,1) params:")
print(f"omega = {omega_hat:.6e}, alpha = {alpha_hat:.6f}, beta = {beta_hat:.6f}")
print(f"alpha+beta = {alpha_hat+beta_hat:.6f}")

# compute in-sample sigma2 and volatility
sigma2_in_sample = garch11_sigma2((omega_hat, alpha_hat, beta_hat), returns.values)
vol_in_sample = np.sqrt(sigma2_in_sample) / 100.0  # convert back to fraction (since we scaled returns *100)

# ----------------------------
# 4. Analytic h-step GARCH forecast (deterministic forecast of sigma^2)
#    For GARCH(1,1):
#    sigma_{t+h}^2 = omega * (1 - (alpha+beta)^h) / (1 - (alpha+beta)) + (alpha+beta)^h * sigma_t^2
# ----------------------------
def garch_forecast_analytic(sigma2_t, omega, alpha, beta, steps):
    phi = alpha + beta
    forecasts = np.zeros(steps)
    for h in range(1, steps + 1):
        forecasts[h-1] = omega * (1 - phi**h) / (1 - phi) + (phi**h) * sigma2_t
    return forecasts

last_sigma2 = sigma2_in_sample[-1]
horizon = 30
sigma2_forecast = garch_forecast_analytic(last_sigma2, omega_hat, alpha_hat, beta_hat, horizon)
vol_forecast = np.sqrt(sigma2_forecast) / 100.0  # fraction

# ----------------------------
# 5. Monte Carlo simulation of returns to show distribution of realized vol
#    We'll simulate many paths using the fitted GARCH parameters.
# ----------------------------
def simulate_garch_paths(n_paths, steps, omega, alpha, beta, last_sigma2, random_seed=42):
    rng = np.random.default_rng(random_seed)
    sims = np.zeros((n_paths, steps))
    sigma2_paths = np.zeros((n_paths, steps))
    for p in range(n_paths):
        sigma2 = last_sigma2
        for t in range(steps):
            z = rng.normal()
            sigma2 = omega + alpha * ( (0 if t==0 else sims[p, t-1]) ** 2 ) + beta * sigma2
            # generate return (remember returns were scaled by *100)
            r_t = np.sqrt(sigma2) * z
            sims[p, t] = r_t
            sigma2_paths[p, t] = sigma2
    return sims, sigma2_paths

n_sims = 2000
sims, sigma2_paths = simulate_garch_paths(n_sims, horizon, omega_hat, alpha_hat, beta_hat, last_sigma2)

# Compute realized vol per path (e.g., daily vol); here we show paths' sigma and aggregated percentiles
vol_paths = np.sqrt(sigma2_paths) / 100.0
vol_mean_path = vol_paths.mean(axis=0)
vol_p10 = np.percentile(vol_paths, 10, axis=0)
vol_p90 = np.percentile(vol_paths, 90, axis=0)
vol_p50 = np.percentile(vol_paths, 50, axis=0)

# ----------------------------
# 6. Plot historical volatility + forecast
# ----------------------------
plt.figure(figsize=(14, 6))

# historic (last 300 days for clarity)
window_show = 300
dates = returns.index
historic_dates = dates[-window_show:]
historic_vol = np.sqrt(sigma2_in_sample[-window_show:]) / 100.0

plt.plot(historic_dates, historic_vol, label="Historic GARCH Vol (daily σ)", color='tab:blue')

# forecast dates
last_date = returns.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

plt.plot(forecast_dates, vol_forecast, label="Analytic GARCH Forecast (mean)", linestyle='--', linewidth=2, color='tab:orange')
plt.plot(forecast_dates, vol_p50, label="MonteCarlo median", linestyle='-.', color='tab:green', alpha=0.7)
plt.fill_between(forecast_dates, vol_p10, vol_p90, color='tab:orange', alpha=0.15, label='MC 10-90 pct')

plt.title("Bitcoin Volatility: Historical GARCH(1,1) & 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Daily volatility (std dev, fraction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Print a small summary
# ----------------------------
print("\n30-day analytic forecast (daily volatility as fraction):")
for i, v in enumerate(vol_forecast, 1):
    print(f"Day +{i:2d}: vol = {v:.4%} (MC median {vol_p50[i-1]:.4%}, 10-90pct [{vol_p10[i-1]:.4%}, {vol_p90[i-1]:.4%}])")


#---------------++++++


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download BTC price
btc = yf.download("BTC-USD", start="2017-01-01")["Close"]

# Daily log returns
returns = np.log(btc / btc.shift(1)).dropna()



def garch_11(returns, alpha0=0.000001, alpha1=0.07, beta1=0.92):
    """
    Manual GARCH(1,1), matching formula:
    σ²(t+1) = α0 + α1*r²(t) + β1*σ²(t)
    """
    n = len(returns)
    var = np.zeros(n)
    
    # Initialize variance with unconditional variance
    var[0] = np.var(returns)

    for t in range(1, n):
        var[t] = (
            alpha0 +
            alpha1 * returns.iloc[t-1]**2 +
            beta1 * var[t-1]
        )

    return var


garch_variance = garch_11(returns)
garch_volatility = np.sqrt(garch_variance)  # σ(t)


plt.figure(figsize=(12,5))
plt.plot(garch_volatility, label="GARCH Volatility")
plt.title("Bitcoin Volatility (GARCH(1,1))")
plt.legend()
plt.show()


def forecast_garch_30days(last_return, last_var, alpha0, alpha1, beta1, days=30):
    forecasts = np.zeros(days)

    var_t = last_var

    for i in range(days):
        var_t = alpha0 + alpha1 * last_return**2 + beta1 * var_t
        forecasts[i] = np.sqrt(var_t)

    return forecasts



alpha0 = 0.000001
alpha1 = 0.07
beta1  = 0.92

forecast_30 = forecast_garch_30days(
    returns.iloc[-1],
    garch_variance[-1],
    alpha0, alpha1, beta1
)


plt.figure(figsize=(10,4))
plt.plot(forecast_30)
plt.title("30-Day Bitcoin Volatility Forecast (GARCH(1,1))")
plt.xlabel("Days Ahead")
plt.ylabel("Volatility")
plt.show()


#---------------++++++

#pip install ta-lib

import pandas as pd
import numpy as np
import yfinance as yf

btc = yf.download("BTC-USD", start="2017-01-01")
df = btc.copy()

# --- Indicators ---
df["20W_SMA"] = df["Close"].rolling(20*7).mean()
df["21W_EMA"] = df["Close"].ewm(span=21*7).mean()

# Hull Moving Average (HMA)
def HMA(series, length):
    wma1 = series.rolling(length//2).mean()
    wma2 = series.rolling(length).mean()
    hma = pd.Series.ewm(2*wma1 - wma2, span=int(np.sqrt(length))).mean()
    return hma

df["8W_HMA"] = HMA(df["Close"], 8*7)
df["9W_HMA"] = HMA(df["Close"], 9*7)

# RSI (14)
delta = df["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14).mean()
avg_loss = pd.Series(loss).rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# Auto Fibonacci (simple)
high = df["Close"].rolling(120).max()
low = df["Close"].rolling(120).min()
df["Fib_61"] = low + 0.618 * (high - low)
df["Fib_38"] = low + 0.382 * (high - low)



def garch_x(returns, rsi, sma20, 
            alpha0=0.000001, alpha1=0.07, beta1=0.92,
            gamma_rsi=0.0001, gamma_sma=0.0001):

    n = len(returns)
    var = np.zeros(n)

    var[0] = np.var(returns)

    for t in range(1, n):
        sma_dev = returns.iloc[t] - sma20.iloc[t]  # deviation from SMA
        var[t] = (
            alpha0 +
            alpha1 * returns.iloc[t-1]**2 +
            beta1 * var[t-1] +
            gamma_rsi * rsi.iloc[t-1] +
            gamma_sma * sma_dev
        )

    return var


garchx_var = garch_x(
    returns=df["Close"].pct_change().dropna(),
    rsi=df["RSI"].fillna(50),
    sma20=df["20W_SMA"].fillna(df["Close"].mean())
)

garchx_vol = np.sqrt(garchx_var)


plt.figure(figsize=(12,5))
plt.plot(garch_volatility, label="Standard GARCH")
plt.plot(garchx_vol, label="GARCH-X (RSI + SMA)")
plt.legend()
plt.show()

def garch_x_forecast(last_return, last_var, rsi, sma_dev,
                     alpha0, alpha1, beta1,
                     gamma_rsi, gamma_sma,
                     steps=30):

    vol = []
    var_t = last_var

    for i in range(steps):
        var_t = (
            alpha0 +
            alpha1 * last_return**2 +
            beta1 * var_t +
            gamma_rsi * rsi +
            gamma_sma * sma_dev
        )
        vol.append(np.sqrt(var_t))

    return vol


#---------------++++++

#!/usr/bin/env python3
"""
btc_garch_indicators.py

Bitcoin volatility modeling:
- Downloads BTC historical prices (yfinance)
- Computes indicators: 20W SMA, 21W EMA, 8W HMA, 9W HMA, RSI, simple fib levels
- Fits GARCH(1,1) by MLE
- Computes analytic 30-day GARCH forecast
- Monte Carlo simulated forecast paths
- GARCH-X (RSI + SMA deviation) optional
- Plots results and can save CSV outputs

Requirements:
pip install yfinance pandas numpy matplotlib scipy

Run:
python btc_garch_indicators.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# ------------------------
# CONFIG
# ------------------------
SYMBOL = "BTC-USD"
START_DATE = "2015-01-01"
HORIZON = 30  # days to forecast
N_SIMS = 2000  # monte carlo paths
SAVE_CSV = True  # write outputs to CSV
OUTPUT_CSV = "btc_garch_results.csv"
PLOT_SHOW = True  # plt.show()

# ------------------------
# HELPERS: Indicators
# ------------------------
def compute_indicators(df):
    """
    df must have 'Close' column
    Adds:
      - 20W_SMA (20 weeks -> 20*7 days)
      - 21W_EMA
      - 8W_HMA, 9W_HMA
      - RSI (14)
      - Fib_61, Fib_38 (rolling 120 trading days)
    """
    # Weekly windows in trading days approx 7 per week (we use calendar days via rolling by days)
    w_days = 7  # approximate, for weekly-based rolling windows use days multiplier
    df = df.copy()

    # 20W SMA: 20 * 7 = 140 period simple rolling mean
    df["20W_SMA"] = df["Close"].rolling(window=20 * w_days, min_periods=1).mean()

    # 21W EMA: use span = 21*7
    df["21W_EMA"] = df["Close"].ewm(span=21 * w_days, adjust=False).mean()

    # Hull Moving Average (HMA)
    def HMA(series, length):
        length = int(max(1, length))
        half = int(max(1, length // 2))
        sqrt_len = int(max(1, int(np.sqrt(length))))
        wma_half = series.rolling(window=half, min_periods=1).mean()
        wma_full = series.rolling(window=length, min_periods=1).mean()
        raw = 2 * wma_half - wma_full
        hma = raw.rolling(window=sqrt_len, min_periods=1).mean()
        return hma

    df["8W_HMA"] = HMA(df["Close"], 8 * w_days)
    df["9W_HMA"] = HMA(df["Close"], 9 * w_days)

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(window=14, min_periods=1).mean()
    roll_down = loss.rolling(window=14, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # Auto Fibonacci levels using 120-day window
    high = df["Close"].rolling(window=120, min_periods=1).max()
    low = df["Close"].rolling(window=120, min_periods=1).min()
    df["Fib_61"] = low + 0.618 * (high - low)
    df["Fib_38"] = low + 0.382 * (high - low)

    return df

# ------------------------
# GARCH Functions
# ------------------------
def garch11_sigma2(params, rets):
    """
    Given params = [omega, alpha, beta], compute sigma^2 series for GARCH(1,1)
    rets: numpy array (or pd.Series) of returns (same units used in likelihood)
    """
    omega, alpha, beta = params
    T = len(rets)
    sigma2 = np.zeros(T)
    if alpha + beta < 1:
        sigma2[0] = omega / (1 - alpha - beta)
    else:
        sigma2[0] = np.var(rets)
    for t in range(1, T):
        sigma2[t] = omega + alpha * (rets[t - 1] ** 2) + beta * sigma2[t - 1]
    return sigma2

def neg_loglik(params, rets):
    """
    Gaussian log-likelihood for GARCH(1,1)
    returns large penalty if params invalid
    """
    omega, alpha, beta = params
    # positivity constraints
    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e12
    if alpha + beta >= 0.999999:
        return 1e12
    sigma2 = garch11_sigma2(params, rets)
    if np.any(sigma2 <= 0):
        return 1e12
    ll_vec = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (rets ** 2) / sigma2)
    return np.sum(ll_vec)

def fit_garch_mle(rets, init_guess=None):
    """
    Fit GARCH(1,1) by minimizing neg_loglik.
    rets: numpy array
    Returns fitted (omega, alpha, beta), optimization result, sigma2_in_sample
    """
    if init_guess is None:
        # small omega and typical alpha,beta guess
        init_guess = np.array([1e-6, 0.05, 0.9])
    bounds = [(1e-12, None), (0.0, 0.9999), (0.0, 0.9999)]
    cons = ({'type': 'ineq', 'fun': lambda x: 0.9999 - (x[1] + x[2])})
    res = minimize(neg_loglik, x0=init_guess, args=(rets,), bounds=bounds, constraints=cons, method='SLSQP',
                   options={'maxiter': 2000, 'ftol': 1e-8})
    if not res.success:
        print("Optimization warning:", res.message)
    omega_hat, alpha_hat, beta_hat = res.x
    sigma2 = garch11_sigma2(res.x, rets)
    return (omega_hat, alpha_hat, beta_hat), res, sigma2

def garch_forecast_analytic(last_sigma2, omega, alpha, beta, steps):
    """Analytic h-step deterministic forecast for GARCH(1,1)."""
    phi = alpha + beta
    forecasts = np.zeros(steps)
    for h in range(1, steps + 1):
        forecasts[h - 1] = omega * (1 - phi ** h) / (1 - phi) + (phi ** h) * last_sigma2
    return forecasts

def simulate_garch_paths(n_paths, steps, omega, alpha, beta, last_sigma2, random_seed=42):
    """
    Simulate n_paths future returns and sigma2 paths from GARCH(1,1).
    returns:
      - sims: shape (n_paths, steps) of simulated returns (units same as rets used to fit)
      - sigma2_paths: shape (n_paths, steps)
    """
    rng = np.random.default_rng(random_seed)
    sims = np.zeros((n_paths, steps))
    sigma2_paths = np.zeros((n_paths, steps))
    for p in range(n_paths):
        sigma2 = last_sigma2
        prev_r = 0.0
        for t in range(steps):
            # normal shocks z ~ N(0,1)
            z = rng.normal()
            # GARCH recursion uses previous epsilon^2 (prev_r**2) for alpha term.
            sigma2 = omega + alpha * (prev_r ** 2) + beta * sigma2
            r_t = np.sqrt(sigma2) * z
            sims[p, t] = r_t
            sigma2_paths[p, t] = sigma2
            prev_r = r_t
    return sims, sigma2_paths

# ------------------------
# GARCH-X (with exogenous indicators)
# ------------------------
def garch_x_sigma2(params, rets, exog):
    """
    params: vector [omega, alpha, beta, gamma1, gamma2, ...] where exog has matching columns
    rets: numpy array
    exog: 2D array-like shape (T, k)
    Returns sigma2 series.
    Model: sigma_{t} = omega + alpha * re_{t-1}^2 + beta * sigma_{t-1} + sum_j gamma_j * X_{t-1,j}
    (we use t-1 indexing for returns & exog influence to keep causality)
    """
    omega = params[0]
    alpha = params[1]
    beta = params[2]
    gammas = params[3:]
    exog = np.asarray(exog)
    T = len(rets)
    sigma2 = np.zeros(T)
    if alpha + beta < 1:
        sigma2[0] = omega / (1 - alpha - beta)
    else:
        sigma2[0] = np.var(rets)
    for t in range(1, T):
        exog_term = np.dot(gammas, exog[t - 1])
        sigma2[t] = omega + alpha * (rets[t - 1] ** 2) + beta * sigma2[t - 1] + exog_term
        # ensure non-negativity
        if sigma2[t] <= 0:
            sigma2[t] = 1e-12
    return sigma2

# ------------------------
# MAIN: download, compute, fit, forecast, plot
# ------------------------
def main():
    print("Downloading BTC data...")
    df = yf.download(SYMBOL, start=START_DATE, progress=False)
    df = df[["Close"]].dropna()
    if df.empty:
        raise RuntimeError("No data downloaded. Check symbol and internet.")

    # indicators
    df = compute_indicators(df)

    # returns: use daily log returns (multiplied by 100 for numeric stability)
    df["logret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    # scale returns by 100 for numerical stability in MLE
    df["ret_scaled"] = df["logret"] * 100.0
    rets = df["ret_scaled"].values

    # Fit GARCH(1,1)
    print("Fitting GARCH(1,1) by MLE ...")
    (omega_hat, alpha_hat, beta_hat), res, sigma2_in_sample = fit_garch_mle(rets)
    print("Fitted params:")
    print(f"omega = {omega_hat:.6e}, alpha = {alpha_hat:.6f}, beta = {beta_hat:.6f}, alpha+beta={alpha_hat+beta_hat:.6f}")

    # convert in-sample variance to original return units (we scaled by 100)
    df["sigma2_garch"] = sigma2_in_sample
    df["vol_garch"] = np.sqrt(df["sigma2_garch"]) / 100.0  # back to fraction

    # Analytic forecast (deterministic expectation for sigma^2) for next HORIZON days
    last_sigma2 = df["sigma2_garch"].iloc[-1]
    sigma2_forecast = garch_forecast_analytic(last_sigma2, omega_hat, alpha_hat, beta_hat, HORIZON)
    vol_forecast = np.sqrt(sigma2_forecast) / 100.0

    # Monte Carlo
    print("Running Monte Carlo simulations ...")
    sims, sigma2_paths = simulate_garch_paths(N_SIMS, HORIZON, omega_hat, alpha_hat, beta_hat, last_sigma2)
    vol_paths = np.sqrt(sigma2_paths) / 100.0  # back to fraction
    vol_p10 = np.percentile(vol_paths, 10, axis=0)
    vol_p50 = np.percentile(vol_paths, 50, axis=0)
    vol_p90 = np.percentile(vol_paths, 90, axis=0)
    vol_mean = vol_paths.mean(axis=0)

    # Optional: Build GARCH-X exogenous inputs (RSI and SMA deviation)
    print("Constructing exogenous inputs for GARCH-X (RSI, SMA dev)...")
    # align exog to df
    df["sma_dev"] = (df["Close"] - df["20W_SMA"]) / df["20W_SMA"]  # relative deviation
    # replace inf/nan
    df["sma_dev"] = df["sma_dev"].fillna(0.0)
    df["RSI_f"] = df["RSI"].fillna(50.0)

    # Example: build exog matrix [RSI_norm, sma_dev]
    # normalize RSI to roughly [-0.5,0.5] by subtracting 50 and dividing by 100
    df["rsi_norm"] = (df["RSI_f"] - 50.0) / 100.0
    exog = df[["rsi_norm", "sma_dev"]].values

    # Fit a simple GARCH-X with two gammas (gamma_rsi and gamma_sma). We'll use simple initial guesses.
    print("Fitting GARCH-X (omega,alpha,beta,g_rsi,g_sma) by quasi-MLE (simple search)...")
    # Define objective for garch-x
    def neg_ll_garch_x(p):
        # impose bounds inside
        omega, alpha, beta = p[0], p[1], p[2]
        gammas = p[3:]
        # basic bounds check
        if omega <= 0 or alpha < 0 or beta < 0:
            return 1e12
        if alpha + beta >= 0.9999:
            return 1e12
        sigma2 = garch_x_sigma2(p, rets, exog)
        if np.any(sigma2 <= 0):
            return 1e12
        ll = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (rets ** 2) / sigma2)
        return np.sum(ll)

    init_gx = np.array([omega_hat, alpha_hat, beta_hat, 0.0001, 0.0001])
    bounds_gx = [(1e-12, None), (0.0, 0.9999), (0.0, 0.9999), (None, None), (None, None)]
    cons_gx = ({'type': 'ineq', 'fun': lambda x: 0.9999 - (x[1] + x[2])})
    try:
        res_gx = minimize(neg_ll_garch_x, x0=init_gx, bounds=bounds_gx, constraints=cons_gx, method='SLSQP',
                          options={'maxiter': 1500, 'ftol': 1e-8})
        if not res_gx.success:
            print("GARCH-X optimization warning:", res_gx.message)
        p_hat_gx = res_gx.x
        print("GARCH-X fitted params:", p_hat_gx)
    except Exception as e:
        print("GARCH-X optimization failed (falling back to small gammas):", e)
        p_hat_gx = np.array([omega_hat, alpha_hat, beta_hat, 0.0, 0.0])

    # compute GARCH-X sigma2 in sample
    sigma2_gx = garch_x_sigma2(p_hat_gx, rets, exog)
    df["sigma2_garch_x"] = sigma2_gx
    df["vol_garch_x"] = np.sqrt(df["sigma2_garch_x"]) / 100.0

    # Forecast GARCH-X for HORIZON days: we must decide how to treat exog future values.
    # Simple choice: hold exog constant at last observed values (rsi_norm_last, sma_dev_last).
    exog_last = exog[-1]
    print("Forecasting GARCH-X with exog held constant (last observed values).")
    omega_gx, alpha_gx, beta_gx = p_hat_gx[0], p_hat_gx[1], p_hat_gx[2]
    gammas_hat = p_hat_gx[3:]

    sigma2_gx_fore = np.zeros(HORIZON)
    var_t = df["sigma2_garch_x"].iloc[-1]
    prev_r = rets[-1]
    for h in range(HORIZON):
        exog_term = np.dot(gammas_hat, exog_last)
        var_t = omega_gx + alpha_gx * (prev_r ** 2) + beta_gx * var_t + exog_term
        sigma2_gx_fore[h] = var_t
        # For deterministic forecast we keep prev_r same as last observed (alternatively zero)
        prev_r = rets[-1]  # or use 0.0
    vol_gx_fore = np.sqrt(sigma2_gx_fore) / 100.0

    # ------------------------
    # Plotting
    # ------------------------
    print("Plotting results...")
    plt.style.use("seaborn-darkgrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)

    # 1) Price + indicators
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    ax.plot(df.index, df["20W_SMA"], label="20W SMA", linewidth=1)
    ax.plot(df.index, df["21W_EMA"], label="21W EMA", linewidth=1)
    ax.plot(df.index, df["8W_HMA"], label="8W HMA", linewidth=1)
    ax.plot(df.index, df["9W_HMA"], label="9W HMA", linewidth=1)
    ax.fill_between(df.index, df["Fib_38"], df["Fib_61"], color="orange", alpha=0.05, label="Fib 38-61")
    ax.set_title(f"{SYMBOL} Price and Indicators")
    ax.legend(loc="upper left")

    # 2) Historical Volatility (GARCH) and GARCH-X
    ax = axes[1]
    ax.plot(df.index, df["vol_garch"], label="GARCH(1,1) vol (daily std)", linewidth=1)
    ax.plot(df.index, df["vol_garch_x"], label="GARCH-X vol", linewidth=1)
    ax.set_title("Estimated Daily Volatility (GARCH & GARCH-X)")
    ax.legend()

    # 3) Forecasted volatility (analytic) + Monte Carlo percentiles + GARCH-X forecast
    ax = axes[2]
    # historic last window
    hist_show = 300
    hist_dates = df.index[-hist_show:]
    ax.plot(hist_dates, df["vol_garch"].iloc[-hist_show:], label="Historic GARCH vol (recent)", color="tab:blue")
    # forecast dates
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    ax.plot(forecast_dates, vol_forecast, label="Analytic GARCH forecast (mean)", linestyle="--", color="tab:orange")
    ax.plot(forecast_dates, vol_p50, label="MC median", linestyle="-.", color="tab:green")
    ax.fill_between(forecast_dates, vol_p10, vol_p90, color="tab:orange", alpha=0.15, label="MC 10-90 pct")
    ax.plot(forecast_dates, vol_gx_fore, label="GARCH-X forecast (held exog)", linestyle=":", color="tab:purple")
    ax.set_title(f"30-day Volatility Forecast (GARCH analytic, Monte Carlo, GARCH-X)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily volatility (std dev, fraction)")
    ax.legend()

    plt.tight_layout()
    if PLOT_SHOW:
        plt.show()

    # ------------------------
    # Save outputs to CSV
    # ------------------------
    if SAVE_CSV:
        print("Saving results to CSV:", OUTPUT_CSV)
        # prepare forecast dataframe
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "vol_garch_analytic": vol_forecast,
            "vol_garch_mc_median": vol_p50,
            "vol_garch_mc_p10": vol_p10,
            "vol_garch_mc_p90": vol_p90,
            "vol_garchx_fore": vol_gx_fore
        })
        # combine recent history and forecasts
        out_df = df.copy()
        out_df = out_df.reset_index().rename(columns={"index": "date"})
        # Save both in a single CSV with separator between sections
        with open(OUTPUT_CSV, "w") as f:
            out_df.to_csv(f, index=False)
            f.write("\n\n# FORECASTS (next {} days)\n".format(HORIZON))
            forecast_df.to_csv(f, index=False)
        print("Saved.")

    # ------------------------
    # Print short summary of 30-day analytic forecast
    # ------------------------
    print("\n30-day analytic GARCH forecast (daily volatility as fraction):")
    for i, v in enumerate(vol_forecast, 1):
        print(f"Day +{i:2d}: vol = {v:.4%} (MC median {vol_p50[i-1]:.4%}, 10-90pct [{vol_p10[i-1]:.4%}, {vol_p90[i-1]:.4%}])")

    print("\nDone.")

if __name__ == "__main__":
    main()


#---------------++++++
#Date | Close | Return | SMA20W | EMA21W | HMA8 | HMA9 | RSI | FibHigh | FibLow | ...


df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()
df = df.dropna()
#---------------++++++


import numpy as np

def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    MAE = (1/n) * Σ |y_t - ŷ_t|
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    MSE = (1/n) * Σ (y_t - ŷ_t)^2
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE)
    MAPE = (1/n) * Σ |(y_t - ŷ_t) / y_t|
    * Scale independent BUT NOT symmetric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # avoid division by zero
    eps = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    sMAPE = (1/n) * Σ |y_t - ŷ_t| / (|y_t| + |ŷ_t|)
    * Scale independent AND symmetric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred))
    eps = 1e-10
    return np.mean(np.abs(y_true - y_pred) / (denominator + eps)) * 100


y_true = [100, 105, 110, 120]
y_pred = [102, 107, 108, 125]

print("MAE :", mae(y_true, y_pred))
print("MSE :", mse(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))
print("sMAPE:", smape(y_true, y_pred))


#---------------++++++

# 1) Fit GARCH-X -> produces sigma_hat series
# use arch.arch_model(..., mean='Zero', vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
# supply exog = np.column_stack([rsi, sma_dev, ...]) to fit()

# 2) Build features
X = pd.DataFrame({
    'ret': rets,
    'sma20w': df['20W_SMA_dev'],
    'rsi': df['RSI'],
    'sigma_hat': sigma_hat,           # from GARCH-X
    'sigma_30d': sigma30,             # analytic or smoothed forecast
    # ... other indicators
})
y = (df['ret'].shift(-1) > 0).astype(int)

# 3) TimeSeries split, train XGBoost classifier
from xgboost import XGBClassifier
model = XGBClassifier(...)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)



#---------------++++++

#pip install yfinance pandas numpy matplotlib scipy arch xgboost scikit-learn

#!/usr/bin/env python3
"""
btc_garch_xgboost_pipeline.py

Full pipeline:
- Download BTC daily prices
- Compute weekly indicators (20W SMA, 21W EMA, 8/9W HMA), RSI, auto-Fib
- Fit GARCH(1,1) via arch package (Student-t supported) -> volatility features
- Optionally fit a custom GARCH-X (variance exogenous) via MLE (if desired)
- Build features and target for price direction (next-day up/down)
- Train XGBoost classifier using TimeSeriesSplit
- Evaluate metrics and run a simple backtest
- Compute volatility forecast metrics (MAE/MSE/MAPE/sMAPE)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import time

# ------------------------
# CONFIG
# ------------------------
SYMBOL = "BTC-USD"
START_DATE = "2016-01-01"
HORIZON = 30
N_SIMS = 1000  # Monte Carlo for volatility paths (optional)
TEST_SIZE_DAYS = 365  # last N days used as holdout for final evaluation/backtest
RANDOM_STATE = 42
PLOT = True

# ------------------------
# Utility metrics (MAE/MSE/MAPE/sMAPE)
# ------------------------
def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-10
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

# ------------------------
# Indicators
# ------------------------
def HMA(series, length):
    length = int(max(1, length))
    half = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))
    wma_half = series.rolling(window=half, min_periods=1).mean()
    wma_full = series.rolling(window=length, min_periods=1).mean()
    raw = 2 * wma_half - wma_full
    hma = raw.rolling(window=sqrt_len, min_periods=1).mean()
    return hma

def compute_indicators(df):
    df = df.copy()
    # approximate: 1 week ≈ 7 days
    w = 7
    df["20W_SMA"] = df["Close"].rolling(window=20*w, min_periods=1).mean()
    df["21W_EMA"] = df["Close"].ewm(span=21*w, adjust=False).mean()
    df["8W_HMA"] = HMA(df["Close"], 8*w)
    df["9W_HMA"] = HMA(df["Close"], 9*w)

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(window=14, min_periods=1).mean()
    roll_down = loss.rolling(window=14, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50.0)

    # Auto Fibonacci using 120-day lookback
    high = df["Close"].rolling(window=120, min_periods=1).max()
    low = df["Close"].rolling(window=120, min_periods=1).min()
    df["Fib_618"] = low + 0.618 * (high - low)
    df["Fib_382"] = low + 0.382 * (high - low)

    # SMA deviation (relative)
    df["SMA20W_dev"] = (df["Close"] - df["20W_SMA"]) / df["20W_SMA"]

    return df

# ------------------------
# Fit GARCH(1,1) via arch package
# ------------------------
def fit_garch_arch(rets, dist="t"):
    """
    Fit GARCH(1,1) to rets (pd.Series). Returns fitted model result and sigma2 in-sample.
    use dist='Normal' or 't' for Student-t.
    """
    # arch expects a numpy array or series
    am = arch_model(rets * 100.0, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist=dist)  # scale returns by 100 for numerics
    res = am.fit(disp="off")
    sigma2 = res.conditional_volatility ** 2  # conditional_volatility is sigma(t)
    # Note: sigma2 in same units as (rets*100)^2 -> convert when used
    return res, sigma2.values

# ------------------------
# Optional: Custom GARCH-X MLE (variance exog) - simple implementation
# Note: arch library allows exog in the mean, but not in variance; this custom MLE is a simple example.
# ------------------------
def garch_x_sigma2(params, rets, exog):
    # params = [omega, alpha, beta, gamma1, gamma2, ...]
    omega = params[0]; alpha = params[1]; beta = params[2]
    gammas = np.array(params[3:])
    T = len(rets)
    sigma2 = np.zeros(T)
    if alpha + beta < 1:
        sigma2[0] = omega / (1 - alpha - beta)
    else:
        sigma2[0] = np.var(rets)
    for t in range(1, T):
        ex = np.dot(gammas, exog[t-1])
        sigma2[t] = omega + alpha * (rets[t-1] ** 2) + beta * sigma2[t-1] + ex
        if sigma2[t] <= 0:
            sigma2[t] = 1e-12
    return sigma2

def neg_loglik_garch_x(params, rets, exog):
    if params[0] <= 0 or params[1] < 0 or params[2] < 0:
        return 1e12
    if params[1] + params[2] >= 0.9999:
        return 1e12
    sigma2 = garch_x_sigma2(params, rets, exog)
    if np.any(sigma2 <= 0):
        return 1e12
    ll = 0.5 * (np.log(2*np.pi) + np.log(sigma2) + (rets**2)/sigma2)
    return np.sum(ll)

def fit_garch_x_custom(rets, exog, init=None):
    """
    Fit GARCH-X with exog in variance via scipy.minimize.
    rets: numpy array of returns (scaled same as fitting)
    exog: array shape (T, k)
    returns params and sigma2
    """
    T, k = len(rets), exog.shape[1]
    if init is None:
        init = np.concatenate([[1e-6, 0.05, 0.9], np.zeros(k)])
    bounds = [(1e-12, None), (0.0, 0.9999), (0.0, 0.9999)] + [(-1.0, 1.0)] * k
    cons = ({'type': 'ineq', 'fun': lambda x: 0.9999 - (x[1] + x[2])})
    res = minimize(neg_loglik_garch_x, x0=init, args=(rets, exog), bounds=bounds, constraints=cons, method='SLSQP',
                   options={'maxiter':1000, 'ftol':1e-8})
    if not res.success:
        print("GARCH-X MLE warning:", res.message)
    params = res.x
    sigma2 = garch_x_sigma2(params, rets, exog)
    return params, sigma2

# ------------------------
# Build features & target
# ------------------------
def build_features(df):
    df = df.copy()
    # daily log returns (not scaled)
    df["logret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    # use scaled returns for GARCH fit numeric stability
    df["ret_scaled"] = df["logret"] * 100.0

    # direction target: next-day price up (1) or not (0)
    df["target_dir"] = (df["logret"].shift(-1) > 0).astype(int)

    # candidate features
    # price-based and indicators
    df["close_lag1"] = df["Close"].shift(1)
    df["ret_lag1"] = df["logret"].shift(1)
    df["sma20w_dev"] = df["SMA20W_dev"]
    df["rsi"] = df["RSI"]
    df["hma8"] = df["8W_HMA"]
    df["hma9"] = df["9W_HMA"]
    df["fib618_dev"] = (df["Close"] - df["Fib_618"]) / df["Fib_618"]

    # fill missing indicator values carefully (forward fill then backfill)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(subset=["ret_scaled", "target_dir"])  # ensure we have core columns

    return df

# ------------------------
# Train XGBoost via time-series CV
# ------------------------
def train_xgb_time_series(X, y, n_splits=5, random_state=RANDOM_STATE):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    val_scores = []
    split = 0
    for train_idx, val_idx in tscv.split(X):
        split += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state
        )
        model.fit(X_train, y_train,
                  early_stopping_rounds=30,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        val_scores.append(acc)
        models.append(model)
        print(f"Split {split}/{n_splits} validation accuracy: {acc:.4f}")
    print("CV mean accuracy:", np.mean(val_scores))
    # return last model for final predictions plus all models if needed
    return models[-1], models

# ------------------------
# Simple backtester (long-only)
# ------------------------
def backtest_signal(df, signal_col="signal", return_col="logret"):
    """
    Simple backtest:
    - If signal==1 => go long 1 unit of BTC for next day; else hold cash.
    - No leverage and no transaction costs (you can add them)
    Returns cumulative returns series and summary metrics.
    """
    df = df.copy()
    df["strategy_ret"] = df[signal_col].shift(0) * df[return_col]  # trade uses today's signal for today's return (alternatively shift)
    df["strategy_ret"].iloc[0] = 0.0
    df["cum_strat"] = (1 + df["strategy_ret"]).cumprod()
    df["cum_buy"] = (1 + df[return_col]).cumprod()
    # performance metrics
    total_ret = df["cum_strat"].iloc[-1] - 1.0
    total_buy = df["cum_buy"].iloc[-1] - 1.0
    ann_ret = (1 + total_ret) ** (252.0 / len(df)) - 1
    # compute daily vol and Sharpe (assume rf=0)
    daily_vol = df["strategy_ret"].std()
    sharpe = (df["strategy_ret"].mean() / (daily_vol + 1e-10)) * np.sqrt(252)
    return df, {"total_ret": total_ret, "ann_return": ann_ret, "sharpe": sharpe, "buy_hold": total_buy}

# ------------------------
# Main pipeline
# ------------------------
def main():
    start_time = time.time()
    print("Downloading BTC data...")
    df = yf.download(SYMBOL, start=START_DATE, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check internet and symbol.")
    df = df[["Close"]].dropna()

    # indicators
    df = compute_indicators(df)

    # build features and target
    df = build_features(df)

    # Fit GARCH(1,1) using arch package to get sigma_t (in-sample)
    print("Fitting GARCH(1,1) via arch...")
    rets_scaled = df["ret_scaled"]
    garch_res, sigma2 = fit_garch_arch(rets_scaled, dist="t")  # Student-t residuals
    # convert sigma2 units: fitted sigma2 corresponds to (ret*100)^2, so convert back to original returns:
    df["sigma2_garch"] = sigma2  # units (scaled returns)^2
    df["vol_garch"] = np.sqrt(df["sigma2_garch"]) / 100.0  # daily vol (fraction)
    df["sigma2_garch_back"] = df["sigma2_garch"] / (100.0**2)  # returns^2 units

    # Optionally fit GARCH-X custom (exog in variance)
    print("Fitting custom GARCH-X (variance exogenous) - optional...")
    # Exog choices: rsi_norm and sma_dev
    exog = np.vstack([ (df["RSI"] - 50.0)/100.0, df["sma20w_dev"] ]).T
    try:
        params_gx, sigma2_gx = fit_garch_x_custom(df["ret_scaled"].values, exog)
        df["sigma2_garch_x_custom"] = sigma2_gx
        df["vol_garch_x_custom"] = np.sqrt(df["sigma2_garch_x_custom"]) / 100.0
        print("Custom GARCH-X fitted.")
    except Exception as e:
        print("Custom GARCH-X fit failed:", e)
        df["sigma2_garch_x_custom"] = df["sigma2_garch"]
        df["vol_garch_x_custom"] = df["vol_garch"]

    # build model feature matrix for classifier
    features = [
        "ret_lag1", "sma20w_dev", "rsi", "hma8", "hma9", "fib618_dev",
        "vol_garch", "vol_garch_x_custom"
    ]
    X = df[features].copy().dropna()
    y = df.loc[X.index, "target_dir"]

    # split into train and test with time
    cutoff = X.index[-TEST_SIZE_DAYS] if TEST_SIZE_DAYS < len(X) else X.index[int(len(X)*0.7)]
    X_train = X.loc[:cutoff]
    y_train = y.loc[:cutoff]
    X_test = X.loc[cutoff:]
    y_test = y.loc[cutoff:]

    # standardize features (simple): use train mean/std
    train_mean = X_train.mean()
    train_std = X_train.std().replace(0,1)
    X_train_s = (X_train - train_mean) / train_std
    X_test_s = (X_test - train_mean) / train_std

    # train with time-series CV and return final model
    print("Training XGBoost with TimeSeriesSplit CV...")
    final_model, cv_models = train_xgb_time_series(X_train_s, y_train, n_splits=5)

    # predictions on test
    y_pred_test = final_model.predict(X_test_s)
    y_proba_test = final_model.predict_proba(X_test_s)[:,1]

    # classification metrics
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba_test)
    except:
        auc = np.nan

    print("\nTest Classification Metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # attach signals and backtest on test set
    df.loc[X_test_s.index, "signal"] = y_pred_test
    df.loc[X_test_s.index, "signal_proba"] = y_proba_test
    bt_df, bt_stats = backtest_signal(df.loc[X_test_s.index], signal_col="signal", return_col="logret")
    print("\nBacktest stats (simple long-only):")
    for k,v in bt_stats.items():
        print(f"{k}: {v}")

    # Volatility forecasting errors: compare analytic 1-step ahead sigma2 vs realized r^2
    # For simplicity, compute 1-day-ahead sigma^2 analytic = last sigma2 (persistence), or use model forecast
    # Here we compare in-sample predicted sigma2_garch_back to realized square returns
    realized_r2 = (df["logret"] ** 2).loc[X.index]  # realized r^2
    pred_sigma2 = df["sigma2_garch_back"].loc[X.index]  # predicted variance in returns^2 units
    # convert shapes and drop nans
    mask = (~np.isnan(realized_r2)) & (~np.isnan(pred_sigma2))
    vol_mae = mae(realized_r2[mask], pred_sigma2[mask])
    vol_mse = mse(realized_r2[mask], pred_sigma2[mask])
    vol_mape = mape(realized_r2[mask], pred_sigma2[mask])
    vol_smape = smape(realized_r2[mask], pred_sigma2[mask])
    print("\nVolatility Forecast Error (in returns^2 units):")
    print(f"MAE: {vol_mae:.6e}, MSE: {vol_mse:.6e}, MAPE: {vol_mape:.4f}%, sMAPE: {vol_smape:.4f}%")

    # Plot results
    if PLOT:
        plt.figure(figsize=(14, 10))

        ax1 = plt.subplot(3,1,1)
        ax1.plot(df.index, df["Close"], label="Close")
        ax1.plot(df.index, df["20W_SMA"], label="20W SMA")
        ax1.plot(df.index, df["21W_EMA"], label="21W EMA")
        ax1.set_title("BTC Price + Weekly Indicators")
        ax1.legend()

        ax2 = plt.subplot(3,1,2)
        ax2.plot(df.index, df["vol_garch"], label="GARCH vol (daily σ)")
        ax2.plot(df.index, df["vol_garch_x_custom"], label="GARCH-X custom vol")
        ax2.set_title("Estimated daily volatility (σ)")
        ax2.legend()

        ax3 = plt.subplot(3,1,3)
        # plot test period actual returns sign vs predicted signals
        test_idx = X_test_s.index
        ax3.plot(test_idx, df.loc[test_idx, "logret"], label="log returns", alpha=0.6)
        ax3.plot(test_idx, df.loc[test_idx, "signal_proba"], label="signal prob (pred up)", color="C1", alpha=0.9)
        ax3.set_title("Test returns and predicted probability (signal)")
        ax3.legend()

        plt.tight_layout()
        plt.show()

    end_time = time.time()
    print(f"\nDone. Elapsed time: {end_time - start_time:.1f} s")

if __name__ == "__main__":
    main()


#---------------++++++

from arch import arch_model
import pandas as pd

# r = returns
# X = dataframe of exogenous indicators (e.g., RSI, HMA, SMA20W)

model = arch_model(
    r,
    vol='GARCH',
    p=1,
    q=1,
    mean='ARX',        # <-- EXOGENOUS VARIABLES HERE
    lags=0,
    x=X,               # <-- your indicators
    dist='t'           # <-- robust heavy-tail distribution (recommended for Bitcoin)
)

res = model.fit(update_freq=5)
print(res.summary())
