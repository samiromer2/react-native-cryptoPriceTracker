Skip to content
Chat history
You said:
can you make this into python code
ChatGPT said:

Sure! Here’s clean, simple Python code for both formulas shown in your image.
1. Sample Variance (using 1n−1∑(xi−xˉ)2n−11​∑(xi​−xˉ)2)

def sample_variance(x):
    n = len(x)
    mean_x = sum(x) / n
    return sum((xi - mean_x) ** 2 for xi in x) / (n - 1)

2. Population Variance (using E[(x−E(x))2]E[(x−E(x))2])

def population_variance(x):
    n = len(x)
    mean_x = sum(x) / n
    return sum((xi - mean_x) ** 2 for xi in x) / n

Usage Example

data = [2, 4, 6, 8]

print("Sample variance:", sample_variance(data))
print("Population variance:", population_variance(data))

If you want a NumPy version too, tell me!
You said:
this is conditinal plz do the same , convert this into python code
ChatGPT said:

Here you go — clean Python code that matches the formula in your image:
Var(x∣I)=σcond2=E[(x−E(x∣I))2]
Var(x∣I)=σcond2​=E[(x−E(x∣I))2]

This means:
the variance of xx given some condition II is the expected squared deviation from the conditional mean.
✅ Python version (general, flexible)

You pass in:

    x: your data values

    I: a Boolean mask or condition selecting the subset where condition holds

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

✅ Example usage

x = [2, 4, 6, 8, 10]
I = [True, False, True, False, True]  # Condition

print("Conditional variance:", conditional_variance(x, I))

Want a NumPy version or a version where I is a function?

I can give you both if you want!
You said:
Bitcoin Price Data
Mean Type	Primarily Conditional Mean (with some influence from Unconditional Mean)
so change the code plz
ChatGPT said:

Got it — you want a hybrid mean:
✔️ Primarily Conditional Mean

but with
✔️ Some Influence From the Unconditional Mean

This is extremely common in Bayesian statistics and signal smoothing (like Kalman filters).

So we define:
μ=α E(x∣I)  +  (1−α) E(x)
μ=αE(x∣I)+(1−α)E(x)

Where:

    α≈0.7−0.95α≈0.7−0.95 → mostly conditional

    E(x∣I)E(x∣I) = conditional mean

    E(x)E(x) = unconditional (global) mean

Then conditional variance becomes:
Var(x∣I)=E[(x−μ)2∣I]
Var(x∣I)=E[(x−μ)2∣I]
✅ Updated Python Code (Hybrid Mean Version)

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

✅ Example Usage

x = [20000, 20500, 21000, 22000, 18000, 19500]  # Bitcoin prices
I = [True, True, False, False, True, False]     # e.g., "during high volume hours"

variance = conditional_variance_hybrid_mean(x, I, alpha=0.9)
print("Conditional Variance (Hybrid Mean):", variance)

⭐ Why this fits Bitcoin Price Data

Bitcoin price volatility is influenced by:

    Recent market conditions (conditional)

    Long-term baseline trend (unconditional)

So using:

    α=0.85α=0.85 = mostly conditional data

    1−α=0.151−α=0.15 = small global influence

gives more stable, realistic results.

If you want, I can also:

✅ incorporate rolling windows,
✅ use pandas,
✅ automatically compute I using volume or volatility,
✅ build a full Bitcoin volatility model.

Just tell me!
You said:
change this to a python code
ChatGPT said:

Here is the correct Python interpretation of the formula in your image:
Var(εt∣It)=σt2
Var(εt​∣It​)=σt2​

This is time-varying conditional variance — the foundation of GARCH, used in Bitcoin/finance modeling.

Below are three Python versions, depending on what you want.
✅ 1. Basic Python version (direct formula)

If you simply want to compute the conditional variance at time tt:

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

But in finance, Iₜ is usually the past residuals, not the current data point.
✅ 2. Rolling conditional variance (finance-style)

This matches time-series heteroscedasticity:

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

This is similar to:
σt2=Var(εt−window:t∣It)
σt2​=Var(εt−window:t​∣It​)
✅ 3. Actual GARCH(1,1) version (used for Bitcoin volatility)

Since your slide is exactly what GARCH models, here’s the standard:
σt2=ω+αεt−12+βσt−12
σt2​=ω+αεt−12​+βσt−12​

Python:

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

This is the real finance version of your formula.
✔️ Which one do you want to use for Bitcoin price modeling?

    Basic formula (teaching)

    Rolling window conditional variance (simple ARCH)

    Full GARCH(1,1) (industry-standard for BTC volatility)

Tell me and I’ll convert that one into a cleaner function or full example with BTC data.
You said:
ChatGPT said:

Here is the Python code version of the ARCH model shown in your slide.

Your formula:
σt+12=a0+a1rt2
σt+12​=a0​+a1​rt2​

This means:

    Tomorrow’s volatility depends on

        a constant a0a0​

        plus yesterday’s squared return rt2rt2​

This is literally the ARCH(1) model.
✅ Python Implementation of ARCH(1)

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

✅ Example Usage

btc_returns = [0.02, -0.015, 0.01, -0.03, 0.025]

sigma2 = arch_1(btc_returns, a0=0.05, a1=0.7)

print("Conditional variances:", sigma2)

⭐ What the code matches from your slide

✔ ARCH = AutoRegressive Conditional Heteroscedasticity
✔ Volatility changes over time
✔ Captures “volatility clustering”
✔ Uses rt2rt2​ to update future variance
If you want, I can also give you:

✅ Full GARCH(1,1) (more realistic for Bitcoin)
✅ Plotting volatility over time
✅ A version using pandas + NumPy
✅ A version that downloads real Bitcoin historical prices

Just tell me!
You said:
✅ Full GARCH(1,1) (more realistic for Bitcoin)
✅ Plotting volatility over time
✅ A version using pandas + NumPy
✅ A version that downloads real Bitcoin historical prices
ChatGPT said:

Perfect — here is the full package you requested:

✔ Full GARCH(1,1) implementation
✔ Plotting volatility over time
✔ Using pandas + NumPy
✔ Automatically downloads real Bitcoin price data (via yfinance)

This is a complete, ready-to-run script.
🚀 FULL GARCH(1,1) BITCOIN VOLATILITY MODEL (Python)
✅ Install required libraries (if needed)

pip install yfinance pandas numpy matplotlib

📌 Python Script

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

✅ What this script does
1️⃣ Downloads real BTC price data

Uses:

btc = yf.download("BTC-USD")

2️⃣ Computes log returns

Smooth & standard for volatility modeling.
3️⃣ Runs a real GARCH(1,1) model
σt2=ω+αrt−12+βσt−12
σt2​=ω+αrt−12​+βσt−12​

This is the exact formula used in:

    professional finance

    crypto volatility models

    risk estimation

    HFT systems

4️⃣ Plots volatility over time

You will clearly see:
BTC volatility spikes during crashes + bull runs (perfect GARCH behavior).
⭐ BONUS

If you want, I can add:

🔥 GARCH parameter optimization (MLE)
🔥 ARCH vs GARCH comparison
🔥 Volatility forecasting (next 30 days)
🔥 Plot returns vs volatility clustering
🔥 Save results to Excel/CSV

Just tell me “add forecasting” or “add optimizer” and I’ll build it.
You said:
🔥 Volatility forecasting (next 30 days)
ChatGPT said:

Nice — below is a single, ready-to-run Python script that:

    Downloads Bitcoin prices (via yfinance)

    Computes log returns

    Fits a GARCH(1,1) model by Maximum Likelihood (constrained)

    Produces an analytic h-step volatility forecast for the next 30 days (closed-form for GARCH(1,1))

    Runs Monte Carlo simulations to show possible realized volatility paths and uncertainty

    Plots historical volatility and the 30-day forecast (analytic mean) and Monte Carlo percentiles

Install the required packages first if you don’t already have them:

pip install yfinance pandas numpy matplotlib scipy

Then save & run this script:

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

A few notes & tips

    I scaled returns by *100 for numeric stability during optimization; final volatility outputs are converted back to fraction (e.g., 0.02 = 2% daily vol). You can remove scaling if you prefer raw log-return units.

    The analytic h-step formula gives the expected conditional variance for each future day — a deterministic forecast under the fitted model.

    Monte Carlo gives uncertainty (percentiles) by simulating return paths driven by model randomness.

    If alpha + beta is very close to 1, GARCH is nearly nonstationary and forecasts decay slowly; MLE may be unstable — consider longer data or regularization.

    Want MLE improvements? I can:

        add robust starting values,

        add standard errors,

        or switch to arch package’s arch.univariate for battle-tested estimation and tests.

Would you like me to:

    Run the script for you and show the plot & numeric forecast (I can’t run code in your environment, but I can run it here if you ask to execute—note: I can run visible Python code via the notebook tool if you want me to run it in this session), or

    Modify it to save forecasts to CSV / compute annualized volatility / produce a forecast of realized volatility (e.g., next 30-day realized vol)?

You said:
ChatGPT said:

Here is clean Python code that exactly matches the GARCH(1,1) formula shown in your slide:
σt+12=α0+α1rt2+β1σ^t2
σt+12​=α0​+α1​rt2​+β1​σ^t2​

I’ll give you:

✅ Manual GARCH(1,1) implementation (pure NumPy)
✅ Fit & Forecast Bitcoin Volatility (30 days)
✅ Plot volatility over time
✅ Commented so you can teach it easily
✅ 1. Download Bitcoin prices + Compute returns

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download BTC price
btc = yf.download("BTC-USD", start="2017-01-01")["Close"]

# Daily log returns
returns = np.log(btc / btc.shift(1)).dropna()

✅ 2. Manual GARCH(1,1) – Matching the Formula
Formula you want:
σt+12=α0+α1rt2+β1σ^t2
σt+12​=α0​+α1​rt2​+β1​σ^t2​

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

✅ 3. Calculate GARCH Volatility

garch_variance = garch_11(returns)
garch_volatility = np.sqrt(garch_variance)  # σ(t)

📈 Plot Real Bitcoin Volatility (GARCH)

plt.figure(figsize=(12,5))
plt.plot(garch_volatility, label="GARCH Volatility")
plt.title("Bitcoin Volatility (GARCH(1,1))")
plt.legend()
plt.show()

🔥 4. Forecast Next 30 Days Volatility

Using last σ² and last return r²:

def forecast_garch_30days(last_return, last_var, alpha0, alpha1, beta1, days=30):
    forecasts = np.zeros(days)

    var_t = last_var

    for i in range(days):
        var_t = alpha0 + alpha1 * last_return**2 + beta1 * var_t
        forecasts[i] = np.sqrt(var_t)

    return forecasts

Run it:

alpha0 = 0.000001
alpha1 = 0.07
beta1  = 0.92

forecast_30 = forecast_garch_30days(
    returns.iloc[-1],
    garch_variance[-1],
    alpha0, alpha1, beta1
)

📈 Plot 30-Day Volatility Forecast

plt.figure(figsize=(10,4))
plt.plot(forecast_30)
plt.title("30-Day Bitcoin Volatility Forecast (GARCH(1,1))")
plt.xlabel("Days Ahead")
plt.ylabel("Volatility")
plt.show()

✅ Result

You now have:
✔ A GARCH(1,1) model that matches exactly the formula from your slide
✔ Real BTC data
✔ Volatility chart
✔ 30-day forecast
✔ Clean + teachable code

If you want next:

🔥 Fit maximum likelihood parameters (α₀, α₁, β₁)
🔥 Compare GARCH vs ARCH
🔥 Use arch library for full statistical estimation
🔥 Confidence intervals for forecasts

Just tell me.
You said:
can i add those factors , 
Indicators:
*Note: W = Week
- 20W Simple Moving Average
- 21W Exponential Moving Act
- 8 or 9 W Hull Moving Average

Maybe — autofib levels

- Relative Strength Index (RSI)
ChatGPT said:

Yes — you can absolutely integrate those indicators into your GARCH workflow.
There are two ways to add them depending on what you want:
✅ 1. Use indicators as separate analysis (not inside GARCH)

This is most common:
You compute GARCH volatility AND technical indicators, then combine them in charts, signals, dashboards, ML models, etc.
✅ 2. Use indicators inside the GARCH formula (as exogenous variables, GARCH-X)

This is advanced but possible:
σt+12=α0+α1rt2+β1σ^t2+γXt
σt+12​=α0​+α1​rt2​+β1​σ^t2​+γXt​

Where XtXt​ can be:

✔ RSI
✔ Moving averages
✔ Hull MA
✔ Fib deviations
✔ Anything time-aligned

This is called GARCH-X or GARCH with exogenous regressors.
🚀 I will give you both:
-----------------------------------------------------------
✅ PART A — Add the indicators (SMA, EMA, HMA, RSI, Fib)
-----------------------------------------------------------
Install TA-Lib (if available)

pip install ta-lib

If TA-Lib is not available on your system, I will give a pure NumPy fallback too.
✔ Add indicators to Bitcoin data

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

You now have all indicators inside the dataframe.
-----------------------------------------------------------
✅ PART B — Use indicators inside GARCH (GARCH-X)
-----------------------------------------------------------
GARCH-X Formula:
σt+12=α0+α1rt2+β1σt2+γ1RSIt+γ2MA deviation+…
σt+12​=α0​+α1​rt2​+β1​σt2​+γ1​RSIt​+γ2​MA deviation+…

Here is GARCH with RSI + 20W SMA deviation:

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

Apply GARCH-X:

garchx_var = garch_x(
    returns=df["Close"].pct_change().dropna(),
    rsi=df["RSI"].fillna(50),
    sma20=df["20W_SMA"].fillna(df["Close"].mean())
)

garchx_vol = np.sqrt(garchx_var)

-----------------------------------------------------------
📊 PART C — Plot GARCH vs GARCH-X
-----------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(garch_volatility, label="Standard GARCH")
plt.plot(garchx_vol, label="GARCH-X (RSI + SMA)")
plt.legend()
plt.show()

-----------------------------------------------------------
🚀 PART D — Forecast Next 30 Days with Indicators
-----------------------------------------------------------

You can forecast volatility while holding indicators constant or projecting them.

Example:

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

🎉 You now have:
✔ SMA, EMA, HMA
✔ RSI
✔ Auto Fibonacci levels
✔ Standard GARCH(1,1)
✔ GARCH-X with indicators
✔ 30-day volatility forecasting
Want me to combine everything into one clean notebook-style code file?

I can produce:

📘 “Bitcoin Volatility Modeling with GARCH & Technical Indicators — Full Notebook”

Just say yes.
You said:
yes
ChatGPT said:

Great — I will prepare a clean, complete, professional notebook-style file that includes:
📘 Bitcoin Volatility Modeling with GARCH, GARCH-X & Technical Indicators

Full Python notebook:

    Auto-download Bitcoin price data

    Compute returns

    Calculate all key indicators:

        20-Week SMA

        21-Week EMA

        8-Week HMA

        9-Week HMA

        RSI

        Auto Fibonacci bands

    Standard GARCH(1,1)

    GARCH-X (RSI + SMA + HMA + Fib optional)

    30-day volatility forecasting

    Plotting volatility

    Plotting price + indicators

    Plotting forecast vs historical volatility

⚡ The next message I send will contain the full notebook code in one block, ready to paste into Jupyter/VSCode/Python.

Before I generate it:
⭐ Do you want the final notebook in:

    Pure Python (single script .py)

    Jupyter Notebook (.ipynb)

    Both

Which format would you like?
You said:
Pure Python (single script .py)
ChatGPT said:

Nice — here’s a single, self-contained pure Python script you can save as btc_garch_indicators.py and run in a local environment (or in a notebook cell). It:

    downloads BTC prices via yfinance

    computes indicators: 20W SMA, 21W EMA, 8W & 9W HMA, RSI, simple auto-Fib levels

    computes returns

    fits GARCH(1,1) by MLE (constrained)

    computes analytic 30-day GARCH forecast and Monte Carlo simulated forecast paths (GARCH and GARCH-X option)

    implements GARCH-X (with RSI and SMA dev) and forecasts with exogenous inputs

    plots price + indicators, historical volatility, analytic forecast and Monte Carlo bands

    writes results to CSV if requested

Save and run (requires yfinance, pandas, numpy, matplotlib, scipy). Instructions are at the top of the script.

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

If you want any of these next:

    I can convert the script into a proper Jupyter notebook .ipynb.

    Add command-line arguments (e.g., symbol, start date, horizon).

    Replace the in-house MLE with the arch package for robust estimation and standard errors (recommended for production).

    Add annualized volatility outputs or save plots to PNG.

Which would you like next?
You said:
What specific forecasting model are you currently using or planning to use?
How is your data structured (e.g., daily, hourly)?
ChatGPT said:

Here’s a clear answer based on everything we discussed — short, precise, and tailored to your setup.
✅ Your Forecasting Model (Based on Our Conversation)

You are currently using:
GARCH(1,1) + GARCH-X extensions

A volatility-forecasting model defined as:
σt+12=α0+α1rt2+β1σ^t2
σt+12​=α0​+α1​rt2​+β1​σ^t2​

and you asked to extend it using:

    20-Week SMA

    21-Week EMA

    8/9-Week HMA

    RSI

    Auto Fibonacci levels

So your “planned” model becomes:
GARCH-X (GARCH with exogenous variables)
σt+12=α0+α1rt2+β1σ^t2+γ1RSIt+γ2SMA20Wt+γ3EMA21Wt+γ4HMAt+γ5FibLevelt
σt+12​=α0​+α1​rt2​+β1​σ^t2​+γ1​RSIt​+γ2​SMA20Wt​+γ3​EMA21Wt​+γ4​HMAt​+γ5​FibLevelt​

A more powerful and Bitcoin-realistic model.
✅ Your Data Structure

You are using:
Daily Bitcoin price data

Downloaded as:

    Daily Close

    Daily returns (log or pct)

    Indicators transformed to weekly versions (20W, 21W, 8W, 9W)

So your data frame is:

Date | Close | Return | SMA20W | EMA21W | HMA8 | HMA9 | RSI | FibHigh | FibLow | ...

This is perfect for GARCH-X.

If you'd like, I can also generate:

🔹 A full data schema
🔹 Recommended train-validation-test split
🔹 A full workflow diagram
🔹 The final single .py file integrating everything (as you requested earlier)

Would you like me to generate the full final .py script now?
You said:
What specific weekly indicators are you using?
How do you plan to handle missing values in your data?
What is the frequency of your data (daily, hourly)?
ChatGPT said:

Here are the exact answers based on the model you’re building (and to confirm before generating the final script).
✅ 1. Weekly Indicators You Are Using

You are using these exact weekly indicators:
🔵 Trend Indicators

    20-Week SMA

    21-Week EMA

    8-Week HMA or 9-Week HMA (Hull Moving Average)

🔵 Momentum Indicators

    14-Period RSI (converted to weekly resolution)

🔵 Market Structure Indicators

    Auto Fibonacci Levels

        Weekly swing high

        Weekly swing low

        Fib levels: 0.236, 0.382, 0.5, 0.618, 0.786

✔ These indicators will be included as exogenous variables in GARCH-X.
✅ 2. How You Plan to Handle Missing Values
You are (and should be) doing the following:
A. For indicator warm-up periods (first few rows):

    Use forward fill (fills missing indicator values)

    Then drop any rows where returns are missing

✔ Best practice for time series
✔ Avoids introducing artificial noise
B. For outliers or NaNs from price API:

    Drop rows where Close = 0 or NaN

    Use dropna() after all indicators are calculated

Final cleaning logic (simple & safe):

df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()
df = df.dropna()

✅ 3. Frequency of Your Data

You are using:
Daily Bitcoin price data (1D)

From this, you generate:

    Daily returns

    Weekly indicators (derived from 7D rolling resample)

🔥 Summary (useful for building the final script)
Component	Value
Data Frequency	Daily (1D)
Indicators	20W SMA, 21W EMA, 8/9W HMA, RSI, AutoFib
Model	GARCH(1,1) + Exogenous indicators (GARCH-X)
Missing Values	ffill → bfill → dropna

If this is correct, I can now generate:
✅ One full .py script including:

    Data download

    Weekly indicator generation

    Auto Fibonacci detection

    GARCH-X volatility model

    Next-30-day volatility forecast

    Final return → price projection

    Plots

Just tell me:

➡️ Generate the final full script
or
➡️ Split into modules (data.py, indicators.py, model.py)
You said:
ChatGPT said:

Here is clean Python code implementing all four metrics exactly as shown in your slide:
✅ Python Implementation of MAE, MSE, MAPE, sMAPE

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

✅ Example Usage

y_true = [100, 105, 110, 120]
y_pred = [102, 107, 108, 125]

print("MAE :", mae(y_true, y_pred))
print("MSE :", mse(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))
print("sMAPE:", smape(y_true, y_pred))

🔥 Clean, pure Python, no sklearn — 100% matches your slide formulas.

If you want, I can now integrate these metrics directly into your volatility forecasting GARCH model, so you can evaluate:

    forecast error vs realized volatility

    model accuracy over time

    compare GARCH(1,1) vs GARCH-X

Would you like metrics built into the final forecast script?
You said:
yes plz
ChatGPT said:

Great — I will build these metrics directly into your GARCH forecasting script.

Before I generate the final full script, I must confirm one final thing, so the code fits your dataset exactly.
✅ One final question (important)

➡️ What type of volatility forecasting are you doing?

Choose A or B:
A) Price-based forecasting

You forecast future Bitcoin prices, then compute MAE, MSE, MAPE, sMAPE against actual future prices.

Formula:
Yt=BTC close price
Yt​=BTC close price
B) Volatility forecasting (GARCH)

You forecast future volatility (σ² or σ), then compute error metrics comparing:

    Forecasted volatility σ̂²_t+1
    vs

    Realized volatility r²_t+1 (return squared)

Formula:
Yt=rt2
Yt​=rt2​
🤔 Quick Guide to Choose
Choose A if:

    You want to predict Bitcoin price levels.

Choose B if:

    You want to predict Bitcoin volatility (financial modeling)

    You are using GARCH(1,1), GARCH-X

    You want volatility clustering modeling

    You care about heteroscedasticity

    You want next-30-days volatility forecasts

❤️ Your earlier message suggests B (volatility forecasting)

…but I need your confirmation.
❓ So tell me:

👇 Do you want metrics on:
A) Price forecast?

or
B) Volatility forecast (GARCH)?

Just reply:
A or B
You said:
checking all of them , 
which one , is best of my case .(bitcoin forcasting the price going up or down)
ChatGPT said:

Short answer up front: GARCH alone is not the best tool to predict whether Bitcoin price will go up or down — GARCH models volatility, not direction. For forecasting price direction you’ll get better results by combining a volatility model (GARCH/GARCH-X) as a feature with a supervised classifier/regressor that uses technical indicators and other signals.

Below I’ll explain why, then give a recommended, practical model stack for your Bitcoin use-case and the precise variants I’d try first (including why and how to use them).
Why a pure GARCH isn’t ideal for direction

    GARCH (and its many variants) model the conditional variance σt2σt2​. They tell you how uncertain or volatile returns will be tomorrow, not the sign of returns.

    Price direction depends on drift, momentum, order flow, regime changes, and exogenous events — things GARCH doesn’t capture well.

    That said, volatility is very useful as an input: high volatility periods often change predictive relationships, so use GARCH output as a feature.

Recommended approach (practical & high-performing)

1) Feature engineering

    Daily log returns (target).

    Weekly indicators you specified (20W SMA, 21W EMA, 8/9W HMA).

    RSI (14).

    Auto-Fib deviations, SMA deviations, volume features (if available).

    Volatility feature(s): in-sample GARCH(1,1) or GARCH-X forecast (analytic σ̂_t or Monte Carlo percentile) and realized vol (rolling std).

2) Volatility model (as feature)

    Fit GARCH(1,1) or GARCH-X (include RSI, SMA deviation as exogenous regressors) to produce σ̂_t (daily).

    Use a heavy-tailed error assumption (Student-t) if fitting (crypto returns are fat-tailed): use t-GARCH or arch package with dist='StudentsT'.

    Consider EGARCH or GJR-GARCH if you expect leverage/asymmetry (negative shocks cause different volatility response).

3) Direction model (supervised)

    Use the features + volatility features to predict direction:

        Binary classification: sign(return_{t+1}) (up/down)

        Or regression: predict next return, then threshold for direction.

    Models to try (in order):

        Gradient boosting (XGBoost / LightGBM / CatBoost) — often best baseline for tabular features.

        Logistic regression with regularization — interpretable baseline.

        Random Forest — robust baseline.

        Neural nets / LSTM — if you have long sequences and lots of data, but more fragile.

    Use GARCH/GARCH-X output as one or more features (current σ̂, 7-day forecast, MC median/percentiles).

4) Train / validation strategy

    Use time-series cross-validation (rolling / expanding window) — no random shuffle.

    Backtest with transaction costs and simple position sizing (e.g., long if predicted up, short if allowed).

5) Evaluation

    For direction: accuracy, precision/recall, ROC-AUC, and profitability metrics (Sharpe, return, max drawdown) with simulated trading.

    For volatility submodels: MSE or sMAPE vs realized r².

Which specific GARCH variants to try (ordered)

If you still want to choose among the many GARCH types, here’s what to try first for Bitcoin:

    GARCH(1,1) (baseline)

        Simple, robust, useful to produce σ̂_t feature.

    GARCH-X (GARCH with exogenous indicators)

        Add RSI, SMA deviation, HMA dev — gives you the best of both: volatility + indicators influence.

    t-GARCH / GARCH with Student-t errors

        Fit with Student-t residuals to handle fat tails; good for crypto.

    EGARCH or GJR-GARCH

        If you find asymmetry (negative shocks affect vol differently), try these.

    FIGARCH or HYGARCH (optional, advanced)

        If you find long-memory in volatility (persistent shocks), consider these. They’re more complex and slower to estimate.

    DCC-GARCH / BEKK (multivariate) (optional)

        If you include multiple assets (BTC + ETH or market indices) to model conditional correlations.

Summary recommendation:
Start with GARCH-X (Student-t) for volatility features, then feed those features into an XGBoost classifier for direction.
How to combine in a single pipeline (concrete)

    Download daily BTC prices → compute returns, indicators.

    Fit GARCH-X on scaled returns (use arch or custom MLE) → produce sigma_hat_t and sigma_hat_horizon (30-day analytic forecast).

    Build feature table for each date t: [{Close_t, ret_t, SMA20W_t, EMA21W_t, HMA8_t, RSI_t, Fib_dev_t, sigma_hat_t, sigma_forecast_30d, ...}]

    Target: y = 1 if return_{t+1} > 0 else 0 (or > threshold).

    Train XGBoost with time-series CV. Evaluate metrics and backtest strategy.

    If profitable/robust, tune, add risk management signals.

Tooling / packages I recommend

    pandas, numpy, matplotlib

    yfinance (data)

    arch (robust GARCH estimation; supports Student-t, EGARCH, etc.)

    scikit-learn (preprocessing, CV)

    xgboost or lightgbm (classifier)

    joblib (parallelization)

Quick code sketch (conceptual) — combine GARCH-X + XGBoost

(very short; I can expand into full runnable script)

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

If you want, I will:

    produce the full .py script that implements this entire pipeline end-to-end (data, indicators, GARCH-X fit via arch, XGBoost classification, time-series CV, backtest and evaluation metrics), or

    generate a simpler starter notebook with the minimal pipeline (GARCH-X → XGBoost) you can iterate on.

Which would you prefer?
You said:
produce the full .py script that implements this entire pipeline end-to-end (data, indicators, GARCH-X fit via arch, XGBoost classification, time-series CV, backtest and evaluation metrics), or

generate a simpler starter notebook with the minimal pipeline (GARCH-X → XGBoost) you can iterate on.
ChatGPT said:

Perfect — below is a single, ready-to-run Python script that implements the full pipeline you requested:

    Data download (yfinance)

    Indicator computation (20W SMA, 21W EMA, 8/9W HMA, RSI, auto-Fib)

    Fit GARCH(1,1) with arch to produce volatility features (Student-t optional)

    (Optional) Fit a simple GARCH-X via custom MLE for variance exogenous regressors (if you want exog in variance)

    Build features and target for direction (next-day up/down)

    Train XGBoost classifier using time-series cross validation

    Evaluate classification metrics (accuracy, precision, recall, F1, ROC AUC)

    Build a simple backtest (long when model predicts up, flat otherwise) and compute performance stats

    Compute volatility forecast metrics (MAE, MSE, MAPE, sMAPE) comparing forecasted σ² vs realized r²

Save as btc_garch_xgboost_pipeline.py and run.

Install required packages first:

pip install yfinance pandas numpy matplotlib scipy arch xgboost scikit-learn

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

Notes & suggestions

    arch supports exogenous regressors in the mean, not directly in the variance. I included an optional custom GARCH-X MLE routine (simple) that places exogenous variables inside the variance equation; it uses scipy.minimize and may be slower/less robust than production implementations.

    For robust GARCH-X estimation and statistical inference, consider R packages (e.g., GARCH.X) or specialized econometrics libraries. For production, the arch package with exog in mean + using the GARCH(1,1) σ̂ as a feature is a strong, robust approach.

    The classifier uses a fairly standard XGBoost configuration — tune hyperparameters for your dataset.

    TimeSeriesSplit is used for CV; you can implement custom expanding-window CV to match how you want to backtest.

    Backtest is simple (long only, no transaction costs). Add costs, slippage, position sizing logic to test a trading strategy realistically.

    I used daily data (you asked daily earlier). If you want hourly, change the data source and ensure indicators are computed with appropriate periods.

    If you want a notebook version (.ipynb), or CSV output / saved plots, I can add those.

Would you like me to:

    Add parameter tuning (GridSearch) for XGBoost (time-series-aware)?

    Replace the custom GARCH-X with a more robust estimator (e.g., call arch for mean exog + GARCH and use exog in mean)?

    Add transaction costs, position sizing, and trade logs in the backtest?

You said:
Add parameter tuning (GridSearch) for XGBoost (time-series-aware)?ywa
Add transaction costs, position sizing, and trade logs in the backtest?yes 
Replace the custom GARCH-X with a more robust estimator (e.g., call arch for mean exog + GARCH and use exog in mean)? explain this more plz
ChatGPT said:

Sure — here is a clear explanation of what it means to “replace the custom GARCH-X with a more robust estimator using the arch library (mean model with exogenous variables + GARCH volatility)**.
✅ What “custom GARCH-X” means

Earlier, we were manually constructing:

    A return series

    Weekly indicators (SMA20W, EMA21W, HMA9W, RSI, etc.)

    A custom GARCH-X pipeline like:

σt2=ω+αrt−12+βσt−12+γXt−1
σt2​=ω+αrt−12​+βσt−12​+γXt−1​

Here, X is your extra indicator (an exogenous regressor).

This works, but it’s hand-written and needs:

    manual parameter estimates

    manual volatility recursion

    careful stability checking

    custom optimization (hard!)

✅ Using the arch library gives you a professional GARCH-X

The Python arch library has:
✔ robust ML-based parameter estimation
✔ supports exogenous regressors in the mean equation
✔ robust standard errors
✔ normal or t-distribution residuals

Important detail:
arch does not allow exogenous regressors directly inside the volatility equation.
BUT it does allow exogenous variables in the mean equation, which is the standard form of GARCH-X in finance literature.

This is the model form:
✅ Professional “GARCH-X” using exogenous variables in mean equation

The model becomes:
📌 Mean model
rt=μ+θXt+ϵt
rt​=μ+θXt​+ϵt​

Where:

    XtXt​ are your indicators (RSI, MA20W, HMA9W, etc.)

    ϵtϵt​ is the residual (innovation)

📌 Variance model (GARCH(1,1))
σt2=ω+αϵt−12+βσt−12
σt2​=ω+αϵt−12​+βσt−12​
✔ This is EXACTLY what is used in:

    Bitcoin volatility papers

    Commodity price forecasting

    High-frequency volatility modelling

    Cross-asset volatility spillovers

🔥 Why this is better than a custom GARCH-X?
Feature	Custom GARCH-X	arch GARCH-X
Parameter Estimation	manual	ML optimization
Stability Check	manual	automatic
Error Distribution	normal only	normal, t-dist, skew-t
Exogenous Variables	manual	built-in (mean model)
Robustness	low	industry standard

This is mathematically more stable and gives better forecasts.
📌 Example of Robust GARCH-X using arch

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

This gives you:

    coefficients for MA/EMA/HMA/RSI

    estimated GARCH parameters

    volatility forecasts

    standard errors

    diagnostics

📌 Why exogenous variables go into the mean (not volatility)?

The accepted academic definition of GARCH-X is:

X modifies the conditional mean, while the volatility is still GARCH.

Most GARCH-X papers use:

    exogenous variables affect returns via forecasting power

    volatility driven by past squared shocks and past volatility

This is more stable and avoids:

    negative variance issues

    overfitting the volatility equation

    identifiability problems

🔥 How this fits your full pipeline
1️⃣ Compute indicators (RSI, HMA, SMA20W, EMA21W)
2️⃣ Fit a robust GARCH-X (exogenous indicators → mean)
3️⃣ Extract:

    volatility forecast

    mean return forecast

    residual features

4️⃣ Train XGBoost to predict direction
5️⃣ Time-series cross validation
6️⃣ Backtest with:

    transaction costs

    position size rules

    PnL curve

    trade logs

If you want…

I can now generate:
✅ The full .py script combining:

    data loading

    indicator calculation

    robust GARCH-X fit

    XGBoost with time-series GridSearch

    backtest engine (fees + trade logs)

    evaluation metrics

    final volatility + directional forecast

or:
📓 A minimal starter notebook version
🔥 What do you prefer?
A) Full production-level Python script?

Contains everything end-to-end.
B) A clean Jupyter notebook starter pipeline?

Tell me A or B.

hat
ChatGPT can make mistakes. Check important info.
