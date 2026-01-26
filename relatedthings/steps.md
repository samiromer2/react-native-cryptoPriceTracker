To start, build two blocks:

a Bitcoin volatility model with EGARCH/GARCH, and

a direction (up/down) classifier that uses that volatility as a feature.​

Step 1: Get BTC data and returns
Use daily BTC-USD data from Yahoo Finance via yfinance, like the Medium article but with Bitcoin instead of stocks.​

Download several years of "BTC-USD" daily prices.

Compute log returns 
r
t
=
ln
⁡
(
P
t
/
P
t
−
1
)
r 
t
 =ln(P 
t
 /P 
t−1
 ); these returns go into the GARCH model and also become features/targets for the classifier.​

Step 2: Fit an EGARCH/GARCH model
Following the article’s structure and crypto papers, fit an EGARCH(1,1) or another GARCH‑family model on the return series using arch.​

Check that returns are stationary and have ARCH effects (the article does this for traditional assets; crypto papers show the same is true for BTC).​

Estimate the model and generate one‑step‑ahead conditional volatility forecasts 
σ
^
t
+
1
σ
^
  
t+1
  for each day; save these as a new time series.​

Step 3: Build up/down labels and features
Turn your problem into: “Will next day’s return be positive?”​

Define label 
y
t
+
1
=
1
y 
t+1
 =1 if 
r
t
+
1
>
0
r 
t+1
 >0, else 
0
0.

Create features at time 
t
t: recent lagged returns, moving averages, RSI or other indicators, volume if available, and the GARCH volatility forecast 
σ
^
t
+
1
σ
^
  
t+1
 .​

Step 4: Train a classifier
Use a supervised model to map features at time 
t
t to label 
y
t
+
1
y 
t+1
 .​

Start simple: logistic regression or random forest; later try XGBoost or an LSTM, as used in recent Bitcoin studies.​

Split into train/validation/test by time (no shuffling), and evaluate using accuracy, precision/recall for “up”, and trading‑style metrics (e.g. strategy return assuming you go long when model predicts up).​

Step 5: Iterate and harden the strategy
Once the pipeline runs end‑to‑end, you can iterate.​

Compare different GARCH variants (EGARCH, GJR‑GARCH, TGARCH, CGARCH) as in the crypto volatility papers and see which volatility forecast helps the classifier most.​

Add regime filters (e.g., only trade when predicted volatility is above/below some threshold) and re‑test out‑of‑sample performance.​