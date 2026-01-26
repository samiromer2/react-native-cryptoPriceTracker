How to use this article in your Bitcoin strategy
A good plan is to treat this article as the volatility block in a larger algorithm:

Use the same pipeline but swap the ticker to Bitcoin, e.g. "BTC-USD" from Yahoo Finance, and fit EGARCH/GARCH to the log returns to get daily volatility forecasts.​

Build features for a classifier/regressor:

Recent returns (lags), technical indicators, order‑book or volume features (if you have them).

Volatility forecasts from your EGARCH model as extra features that encode “how risky / explosive” the next period might be.​

Train a model (logistic regression, random forest, gradient boosting, or an LSTM) to predict the sign of next‑day Bitcoin return (up/down), using those features including GARCH‑based volatility.​

Why this is a good starting point
Crypto price‑direction papers explicitly say they use machine learning over price data to predict direction, sometimes also using volatility as an input.​

There is recent work on Bitcoin trading algorithms that integrate technical analysis, macro indicators, sentiment, and ML to generate buy/sell signals; volatility modeling fits naturally into that stack.​