# The-Future-of-Harveston-Attention-Based-Multi-Output-Weather-Forecaster
This project was developed for **"The Future of Harveston"** Kaggle competition (Data Crunch by CodeJam, CSE UOM), a time-series forecasting challenge to predict five environmental variables — average temperature, solar radiation, rainfall, wind speed, and wind direction — across 9 kingdoms (provinces), evaluated using average sMAPE.
## Problem Framing
The dataset is structured as **panel data**: the same set of environmental variables tracked across multiple kingdoms over a decade, with kingdom-specific measurement quirks (e.g. some temperatures recorded in Kelvin) and missing values.

## Approach

### Preprocessing
- Unit normalization (e.g. Kelvin → Celsius)
- Outlier clipping using IQR bounds (1st/99th percentile)
- **Kingdom-wise KNN imputation** for missing values, respecting each region's own data distribution
- Calendar feature engineering (day of year, quarter, month-start/end flags, etc.)
- Lag and rolling-window features per weather variable

### Model Architecture
A single multi-output neural network (not separate per-kingdom models), inspired by ideas from the **Temporal Fusion Transformer (TFT)**:

- Engineered features projected into **5 tokens**, one per weather variable
- **Kingdom identity** encoded via a learned embedding, added to all 5 tokens as a regional conditioning signal
- Stacked **transformer encoder blocks** (multi-head self-attention) — the 5 weather-variable tokens attend to each other, with kingdom context baked in beforehand
- **Wind direction** handled via sin/cos decomposition + angular loss (correctly handles 0°/360° wraparound, which raw-degree sMAPE cannot)
- **Rain amount** predicted via a two-stage gated head (`sigmoid(gate) × relu(magnitude)`) to handle its zero-inflated distribution
- Per-target output heads with physics-appropriate activations (linear/relu/tanh as fitting)
- Combined loss: sMAPE (4 main targets) + angular loss (wind direction), weighted

### Train/Validation Split
Kingdom-wise time-based 80/20 split — for each kingdom independently, the earliest 80% of dates form training data and the most recent 20% form validation, preventing temporal leakage.

## Results
- Top-10 placement on the leaderboard
- Contributed to team's **2nd place finish** in Phase 1
- Kingdom embedding allowed one shared model to adapt to regional differences without training separate per-kingdom models

## What I'd Explore Next
- Baseline comparison against gradient-boosted models (LightGBM/XGBoost) on the same engineered features — tabular weather data often favors tree ensembles
- Kingdom-wise sMAPE breakdowns to identify regions with systematically higher error
- Direct comparison against fully separate per-kingdom models to evaluate whether shared attention/embedding is worth its added complexity

## Tech Stack
- Python, TensorFlow/Keras, pandas, scikit-learn, matplotlib/seaborn

## Files
- `Harveston_Attention_v3.ipynb` — full pipeline: preprocessing, feature engineering, model, training, evaluation, residual analysis
