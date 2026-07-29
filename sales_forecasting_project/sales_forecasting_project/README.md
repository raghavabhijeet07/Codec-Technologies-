# Sales Forecasting — Data Analytics Internship Project

Predicting future e-commerce sales from historical daily sales data using
three complementary forecasting approaches: **SARIMA**, **Prophet**, and
**LSTM**. Built to include trend, weekly/yearly seasonality, and promotional
effects, with forecast-vs-actual visualizations for each model.

## Project structure

```
sales_forecasting_project/
├── data/
│   ├── generate_data.py       # Creates the synthetic sales dataset
│   └── sales_data.csv         # Generated dataset (created on first run)
├── models/
│   ├── arima_model.py         # SARIMA model (statsmodels + pmdarima)
│   ├── prophet_model.py       # Prophet model (Meta/Facebook Prophet)
│   └── lstm_model.py          # LSTM model (TensorFlow/Keras)
├── utils/
│   └── visualization.py       # Shared plotting + metrics helpers
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis notebook
├── outputs/                   # Generated plots and metrics (created on run)
├── main.py                    # Runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Dataset

Since this project ships without a specific company's proprietary sales
data, `data/generate_data.py` creates a **realistic synthetic daily
e-commerce sales dataset** (2022–2024, ~1,096 days) with the components
real retail data typically has:

- **Trend** — steady underlying business growth over time
- **Weekly seasonality** — higher sales on Fri/Sat/Sun
- **Yearly seasonality** — smooth seasonal wave + a strong Nov–Dec holiday
  boost
- **Promotions** — fixed campaigns (Black Friday, New Year, mid-year sales)
  plus random flash sales, each with a sales lift
- **Noise** — realistic day-to-day randomness

Columns: `date`, `sales`, `units_sold`, `is_promotion`, `is_holiday_season`

> **Using your own data instead:** if your internship gives you a real
> dataset (e.g. a Kaggle retail/e-commerce dataset, or company POS data),
> just replace `data/sales_data.csv` with your file, keeping a `date` column
> and a `sales` column (rename yours if needed). Everything downstream
> works unchanged as long as those two columns exist. If you have an
> `is_promotion` flag, Prophet will automatically use it as a regressor.

## Models

| Model | Library | How it handles seasonality/promos |
|---|---|---|
| **SARIMA** | `statsmodels` / `pmdarima` | `auto_arima` searches `(p,d,q)(P,D,Q,7)` — the `7` captures weekly seasonality directly |
| **Prophet** | `prophet` | Automatic yearly + weekly seasonality decomposition; `is_promotion` added as an extra regressor to learn the promo lift |
| **LSTM** | `tensorflow`/`keras` | Sliding 30-day window over `[sales, is_promotion, is_holiday_season]`, 2-layer LSTM, recursive multi-step forecasting |

All three models are evaluated on the **same 60-day holdout period** using
**MAE**, **RMSE**, and **MAPE**, so results are directly comparable.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

**Note on Prophet:** the `prophet` package depends on `cmdstanpy`, which
downloads a small C++ backend (`cmdstan`) the first time it's used. This
requires an internet connection on first run but only happens once.

**Note on TensorFlow on Windows:** if plain `tensorflow-cpu` has trouble
installing on your machine, try `pip install tensorflow` instead (the
full package) — it also runs on CPU-only machines, just with a larger
install size.

## Running the project

Run everything (data generation + all 3 models + comparison table) with:

```bash
python main.py
```

Or run models individually:

```bash
python data/generate_data.py     # regenerate the dataset
python models/arima_model.py     # SARIMA only
python models/prophet_model.py   # Prophet only
python models/lstm_model.py      # LSTM only
```

All plots and the metrics CSV are written to `outputs/`:

- `arima_forecast.png`, `prophet_forecast.png`, `lstm_forecast.png` —
  forecast vs. actual for each model
- `prophet_components.png` — Prophet's trend/weekly/yearly decomposition
- `model_comparison_metrics.csv` — MAE/RMSE/MAPE side by side

For exploratory analysis (trend/seasonality/promo charts), open
`notebooks/eda.ipynb` in Jupyter:

```bash
pip install jupyter
jupyter notebook notebooks/eda.ipynb
```

## Sample results

On the synthetic dataset's 60-day holdout, Prophet was the strongest
performer (lowest MAPE), thanks to its native handling of multiplicative
yearly/weekly seasonality and the promotion regressor. SARIMA captured the
weekly seasonality well but lagged on holiday spikes. LSTM was competitive
but, as expected for a fairly small dataset (~1,000 days), didn't
out-perform the statistical models — LSTMs typically need more data or
more feature engineering to beat SARIMA/Prophet on series this size.

*(Exact numbers will vary slightly each time you regenerate the data or
retrain, since there's randomness in both the synthetic data and model
training.)*

## Possible extensions (good for a report's "Future Work" section)

- Add external regressors: marketing spend, competitor pricing, weather
- Hyperparameter tuning (grid search for SARIMA orders, Prophet's
  `changepoint_prior_scale`, LSTM units/layers)
- Ensemble the three models (e.g. weighted average by inverse MAPE)
- Backtesting with rolling-origin cross-validation instead of a single
  holdout split
- Deploy the best model behind a simple API/dashboard (e.g. Streamlit) for
  interactive forecasting
