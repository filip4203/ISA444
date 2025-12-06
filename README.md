# ISA 444 – Hotel Demand Forecasting Project

This repository contains my final project for **ISA 444 (Forecasting)**.  
The goal is to forecast **daily room demand** for a panel of hotels and compare several forecasting methods using time-series cross-validation.

Colab Link: https://colab.research.google.com/drive/1g5qYIaAWSCcHY_dy7XtuJBi5tw5FmIHa?usp=sharing

---

## 1. Project Overview

- **Track chosen:** Option 1 – Hotel Demand Forecasting  
- **Dataset:** `sample_hotels.parquet`
  - 19 hotel properties (`unique_id`)
  - Daily demand (`y`) for each property
  - Forecast horizon: **28 days ahead**  

The project emphasizes:

1. Rigorous model comparison  
2. Time-series cross-validation  
3. Clear communication of results

---

## 2. Methods

### 2.1 Models Implemented

**Baseline statistical models (StatsForecast)**

- **Naive** – forecast equals last observed value.
- **SeasonalNaive** – repeats value from the same day of the last week (season length = 7).
- **AutoETS** – automatic exponential smoothing model with weekly seasonality.

> The original instructions mention AutoARIMA. Due to runtime issues in Colab, I focused on ETS, ML, and neural models, and documented this trade-off.

**Machine learning model (MLForecast + LightGBM)**

- **LGBMRegressor**:
  - Global model (trained on all hotels together)
  - Lags: 1, 7, 14
  - Calendar features: day of week, month

**Neural forecasting models (NeuralForecast)**

- **NBEATS**
- **NHITS**

Both are trained globally across all hotel series with:

- Forecast horizon `h = 28`
- Input window length = 28
- `max_steps = 50` (for practical runtime in Colab)

---

## 3. Cross-Validation Design

- **Type:** Rolling-origin time-series cross-validation
- **Windows:** `N_WINDOWS = 3` (non-overlapping)
- **Horizon:** `H = 28` days
- Each window:
  1. Train models on all history up to cutoff.
  2. Forecast next 28 days.
  3. Compare forecasts to actuals using multiple metrics.

Outputs are stored in `data/cv_results.csv`.

> The project handout suggests 5 CV windows. In this environment, 3 windows represent a compromise between computational cost and robust evaluation.

---

## 4. Evaluation Metrics

For each hotel and each model I compute:

- **ME** – Mean Error  
- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **MAPE** – Mean Absolute Percentage Error  

Metrics are saved in `data/metrics.csv`.

A model “wins” a hotel if it has the **lowest MAE** for that series.  
Model wins are summarized in `data/model_wins.csv`.

---

## 5. Key Results

### 5.1 Average performance (across hotels)

(Approximate average values from `metrics.csv`):

- **NBEATS** – lowest average MAE and RMSE overall.
- **NHITS** and **LGBMRegressor** – close behind NBEATS.
- **AutoETS** – competitive classical baseline.
- **Naive** and **SeasonalNaive** – worst on average, but useful as sanity checks.

### 5.2 Model wins by series

From `model_wins.csv`:

- NBEATS and LGBMRegressor win the largest number of hotel series.  
- Naive and AutoETS occasionally win on very simple or flat series.  
- SeasonalNaive rarely wins but provides a seasonal benchmark.

**Conclusion:**  
A global neural model (NBEATS) is the best single choice on average, with LightGBM as a strong ML alternative.

---

## 6. Final Forecasts

Final 28-day-ahead forecasts for each hotel are created by retraining all models on the full dataset and predicting the next 28 days. These forecasts are stored in:

- `data/final_forecasts.csv`

The notebook also includes a plotting function to visualize history vs. forecasts for any `unique_id`.

---

## 7. Files in This Repository

- `ISA444_Hotel_Forecasting.ipynb` – Main project notebook (Colab).
- `data/final_forecasts.csv` – Final 28-day forecasts for all models and hotels.
- `data/metrics.csv` – ME, MAE, RMSE, and MAPE per model and hotel.
- `data/cv_results.csv` – Full cross-validation predictions.
- `data/model_wins.csv` – Model wins summary by hotel.
- `README.md` – This file.

---

## 8. How to Reproduce

1. Open the notebook in Google Colab.
2. Upload `sample_hotels.parquet` or mount Google Drive and point to the file path.
3. Run all cells:
   - Installs packages
   - Loads and cleans the data
   - Runs cross-validation
   - Computes metrics and model wins
   - Trains final models and generates 28-day forecasts
   - Saves CSV outputs

---

## 9. Conclusion

This project shows that:

- More advanced models (NBEATS, NHITS, LightGBM) can significantly outperform simple baselines on hotel demand data.
- Simple models (Naive, AutoETS) can still be competitive for some individual hotels.
- Rolling cross-validation provides a realistic view of out-of-sample performance.

For a production system, I would choose **NBEATS** as the primary forecasting model, backed up by **LightGBM** and **AutoETS** as interpretable baselines.
