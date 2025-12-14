# cRS 📈

**Relative Strength** quantitative research tool focused on crypto markets, aims to quantify the relation between crypto assets, objectively and dynamically.

**Linear Regression** is the base, using **Exponential Weighted Moving Averages (EWMA)** to mitigate heteroscedasticity and **Ridge** regularization for multicollinearity.

There is a **Hidden Markov Model** integration for regime classification, can be trained to identify all kinds of regimes, including volatility and trend.

The tool is composed by a backend that encapsulates all the logic and a simple **streamlit UI** that reduces the interaction friction.

# Use Cases 🖇️

As a quantitative research tool, this project can be used as a helping hand while analysing the highly correlated, noise full cryptocurrency market. Made to robustify the data in a way that can be interpreted without letting outliers explode estimations.

Beyond exploratory analysis, the tool can be used as a systematic decision-support layer for crypto workflows, quantitative or not:
- Cross-Asset Relative Strength Ranking
- Pair Selection
- Statistical Arbitrage
- Dynamic Beta and Exposure Estimation
- Regime-Aware Signal Conditioning
- Portfolio Construction
- Risk Monitoring
- Correlation Stress Detection
- Research and Model Validation

It must be noted that this tool is not designed for high-frequency or microstructure-driven strategies. Its advantage lies in medium- to low-frequency quantitative research, where robustness, interpretability, and regime awareness matter more than latency.

It is highly advised to have familiarization with the Exponential Weigthed Moving Average as a statistical tool, and its parameteres, mainly the decay factor that controls the responsiveness and memory of the Moving Average.

# Disclaimer ‼️
Not Financial Advice

This project is provided for research and educational purposes only.
Nothing in this repository constitutes investment advice, financial advice, trading advice, or a recommendation to buy or sell any asset.

Cryptocurrency markets are highly volatile and subject to regime shifts, liquidity constraints, and structural risks. Past performance, simulated or historical, is not indicative of future results.

The authors and contributors assume no responsibility for financial losses, damages, or decisions made based on the use of this software.
You are solely responsible for understanding the risks involved and for complying with all applicable laws and regulations in your jurisdiction.

Use at your own risk.

# Installation  🛠️

As a python focused project, it is recommended the use of a environment manager, `ex: conda`

## Requirements

```bash
python = "^3.10"
streamlit = "^1.51"
pandas = "^2.2"
numpy = "^1.26"
statsmodels = "^0.14"
requests = "^2.28"
plotly = "^5.19"
matplotlib = "^3.8"
pydantic-settings = "^2.6"
hmmlearn = "^0.3.3"
scipy = "^1.15.3"
black = "^24.3"
ruff = "^0.3"
mypy = "^1.9"
poetry-core = "^1.8.0"
```

## Download or clone the repository main (stable) branch

`git clone https://github.com/jmake1/cRS.git`

## Inside the folder

The project uses poetry to handle all the dependencies and installation requirements.

`pip install .`

It can take a few minutes to fetch everything.

# Run ▶️

The project uses environment variables for certain configurations and paths.

The file .env must be present at the root of the project. Check all the required and optional env variables available in the .env.example file.

The frontend uses streamlit, inside the project folder run:

`streamlit run /src/app/cRS.py`

A webpage should open in your default browser.

Here, the first run can take a while to load, since there are no caches available.

# Main features 🧪

- Historic OHLCV data loading from multiple sources/exchanges

- Data agreggation and treatment

- Dynamic factorization using Linear Regression

- Residual, Beta (factor), alpha (mean), sigma (variance), R2, phi (turnover) and z-score estimation

- Over time correlation visualization

- Hidden Markov Model training and visualization for regime classification

## Data Loader

Data currently can be fetched from **Binance**, **Bybit** and **MEXC** through their public API's, both for spot as well as for perpetuals/futures contracts.

Supported intervals: 1h, 4h, 1D, 1W, 1M.

Data is always read and written to the data folder (env variable, **cRS/data/ by default**). 

The update functionality will **always** check the data available for the respective source, instrument, interval, and update every ticker to the last available timestamp (does a full fetch for not existing tickers, discards not available tickers).

When there is no data available for the combination source + instrument + interval, a full fetch will take place. Can take a few minutes depending on the interval.

The app allows loading from multiple sources and instruments and obtaining a ticker universe `<ticker>_<source>_<instrument>` that can be filtered, the final universe serves as input for the rest of the app.

### Notes
- Memory usage can (and will) escalate very quickly when using a small timeframe + multiple sources + both instruments. Recommended starting interval: 1D.

- Bybit is usually the most reliable source for now, Binance and MEXC can cause issues due to rate limiting/latency/inconsistencies, I am still working on it, and also working on adding more sources/intervals.

## Parameters

- **start_date, end_date**: Data can be filtered by date time range with start_date, end_date.

- **discard, selection**: tickers can be excluded, or a selection of tickers can be set.

- **factors**: Multiple factors are supported, all factor combinations are computed and can be visualized. For example, for the set {BTCUSDT, ETHUSDT}, we have {BTCUSDT}, {ETHUSDT}, {BTCUSDT, ETHUSDT}, to be computed, we can then see the rest of the universe vs each combination as well as BTCUSDT vs ETHUSDT.

- **min_obs**: Minimum number of observations to consider a ticker candidate for analysis.

- **beta_lambda (λβ)**: EWMA parameter, the weigth decaying factor.

## Results

- Factor table with the estimations as of end_date for the factor relationships.

- Universe table with the estimations as of end_date for all tickers in the final universe that are not in the selected factor combination.

- Regression plots for all tickers present in the ticker table: Real return vs Estimated return / Residual return / Cumulative residual return / z-score / Cumulative z-score.

## HMM

**Hidden Markov Chains** are powerfull tools when dealing with time series data with varying regimes. The app integrates a simple implementation of a **Hidden Markov Model** with **Gaussian Emissions** and **Gaussian Mixture Emissions**, from the python library **hmmlearn**.
The model can be trained after the factorization, it uses the computed z-scores that are almost homoscedastic, so we can have better results.

### Model Params

- **Offset** - Wether to discard a number of initial lines of each ticker

- **States** - Number of states to train

- **Mix States** - Only for GMMHMM (Gaussian Mixture Emissions)

- **Iterations** - Number of max iterations

- **Tol** - Tolerance (a value smaller that tol indicates that the model converged)

- **Min Cov** - Minimum of covariance between states

- **Multi** - Wether to give the data resulted from all factor combinations or only from the main combination

- **Type** - HMM / GMMHMM

### Model Visualization

- Plot with state classification for each timestep

- Plot with state probabilities P(state | z_t) for each timestep

- State mean, std, stationarity, duration

- Transition matrix

## Going forward...  ⏩

I will maintain and update the tool with bug fixes, new features, etc as time allows me to. 

Feel free to test, report bugs/problems and suggest ideas to be implemented.

## Contacts  📇

Project related questions or issues can be left in the issues tab.

Other matters -> jmake1@proton.me

I'll try to respond as soon as I can.

## Buy me a coffee  ☕️
- **BTC** - bc1qhwg75jmq99lqjahtsjnpdyl77zlfaynsx0jnpx

- **ETH/BNB/USDT** - 0xb38A9C2C04d891603Ff69A740Ded417EEE42C1F1

- **XMR** - 46ZVTYMZcXPC3UUGyJBkffEFsosydiAvJ4jJxifHVsLoi9z8kDp4kYnWQduFvPmcCZVzX15MfvXSNjVkqQKpVc1RJw3vuP7

- **SOL** - 8mCRUhzuQ7CAJVSWLAZmYsNHMwUrZ1EoVnwTtuc5Ty3g

- **ZEC** - t1TFnj5te7t2VuqEnm4CiZdQWyR5oKzHgQv

- **KAS** - kaspa:qqgw0uujksxhj4lu0emhfjr04adqs6mjth37uvvwcsn8jum5pvhnvyqpfm347

## License

This project is licensed under the Apache License 2.0.
See the LICENSE file for details.








