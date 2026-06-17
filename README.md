# Forecasting Ethereum (ETH/USD) with an LSTM

This project forecasts the price of Ethereum using an LSTM neural network. It is a refactor of an earlier version that, on reflection, never actually *forecast* anything — so the methodology has been rebuilt from the ground up around genuine multi-step forecasting, decomposition, uncertainty and honest backtesting.

## Why this was rebuilt

The original notebook trained an LSTM to predict the **next day's** price from a window of recent prices, then plotted "predicted vs actual" and reported metrics such as R² ≈ 98% and "accuracy" ≈ 98%. Those numbers were misleading. Because the model was given the true price right up to the day before each prediction, the predicted line simply reproduced the actual line shifted by one day — the well-known *parrot problem*. The model looked superb while never producing a forward-looking forecast, and regression "accuracy" is not a meaningful metric.

This version fixes that. The headline changes are:

- **Live data from the CoinCodex open API** (ETH/USD) instead of a static CSV, cached locally for reproducibility, with the original CSV kept only as an offline fallback.
- **STL decomposition** (trend, weekly seasonality, residual) to characterise the series before modelling.
- **Modelling returns, not prices.** Daily log-returns are (near-)stationary, which sidesteps the parrot problem; the price is reconstructed afterwards.
- **A real 6-month forecast** via a recursive/autoregressive rollout — each predicted return is fed back in to produce the next.
- **A 95% confidence interval** from a Monte-Carlo simulation, so the forecast communicates uncertainty rather than a single false-precision line.
- **Walk-forward backtesting** with **MAE** and **MAPE**, in which the LSTM competes against an `auto_arima` statistical baseline and a naive random-walk.

## Methodology

### 1. Data — CoinCodex API
Historical daily ETH/USD closes are pulled from the CoinCodex history endpoint. Because the API down-samples long date ranges, the fetcher requests the history in ≤2-year chunks (each returns roughly daily resolution), concatenates them, resamples to a clean daily series and caches the result to parquet. Subsequent runs read the cache unless `refresh=True`, and if the network is unavailable the code falls back to the bundled CSV.

### 2. STL decomposition (`statsmodels`)
The log-price is decomposed with robust STL using a weekly period (`period=7`). This separates the long multi-year **trend**, the small but persistent weekly **seasonal** component, and the volatility-clustered **residual**, and reports the overall direction of the recent trend.

### 3. Stationarity and returns
The price level is non-stationary (Augmented Dickey-Fuller p ≈ 0.13), whereas daily log-returns are strongly stationary (ADF p ≈ 2e-19). The LSTM is therefore trained on scaled log-returns, and prices are reconstructed by exponentiating the cumulative sum.

### 4. Forecasting and uncertainty
A two-layer LSTM maps a 60-day window of returns to the next return. Forecasting is **recursive**: predict one step, append it, slide the window and repeat for 180 days. The confidence band comes from a Monte-Carlo simulation that combines **MC-dropout** (keeping dropout active at inference to capture model uncertainty) with sampled **return innovations** (to capture the asset's volatility). Taking the 2.5th/50th/97.5th percentiles across the simulated paths yields the median forecast and the 95% interval.


### 5. Walk-forward backtest
Rolling-origin (walk-forward) evaluation is the gold standard for time series. For each fold the model is trained only on data preceding the test block, forecasts 180 days, and is scored against the held-out actuals. The LSTM is compared with `pmdarima.auto_arima` (fit on the same returns) and a naive random-walk.

## Results

From a representative run on history through September 2024 (your live run will differ as new data arrives), averaged over the walk-forward folds:

| Model | MAE (USD) | MAPE |
|---|---|---|
| Naive random-walk | ~528 | ~19.5% |
| `auto_arima` baseline | ~894 | ~35.2% |
| **LSTM (returns, recursive)** | **~823** | **~31.5%** |

Two honest conclusions follow. First, **the LSTM beats the `auto_arima` statistical benchmark** on average — the quantifiable "I beat the baseline" result. Second, **neither model beats a naive random-walk at a 180-day horizon**. That is expected: liquid crypto markets behave close to a random walk, so multi-step *point* forecasts are inherently weak, and the meaningful output of any six-month crypto forecast is the **width of the confidence interval**, not the median line.

## Files

- `main_lstm.ipynb` — the refactored, single-notebook pipeline (data → decomposition → returns → backtest → forecast).
- `ethereum_2015-08-07_2024-09-08.csv` — bundled historical data, used only as an offline fallback.
- `ethereum_price_prediction_lstm.keras` — the trained returns model.
- `metrics.json` — machine-readable summary of the latest run.
- `*.png` — generated decomposition, backtest and forecast charts.
- `main_lstm copy.ipynb` — the original notebook, retained for reference.

## Setup

Python 3.10 is recommended. `numpy<2` is pinned so that `pmdarima` and `tensorflow` remain compatible.

```bash
pip install numpy==1.26.4 pandas scikit-learn statsmodels pmdarima tensorflow-cpu matplotlib pyarrow requests
```

Then open `main_lstm.ipynb` and run the cells top to bottom. On first run it fetches and caches the data; later runs reuse the cache (set `refresh=True` in `fetch_eth_history` to force an update).

## Data credit

Historical ETH/USD price data is provided by the **[CoinCodex](https://coincodex.com) public API**. This project is a methodology demonstration and is **not financial advice**.

## References

- [statsmodels STL documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html)
- [pmdarima (auto_arima) documentation](https://alkaline-ml.com/pmdarima/)
- [TensorFlow / Keras documentation](https://www.tensorflow.org/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [CoinCodex API](https://coincodex.com/page/api/)
