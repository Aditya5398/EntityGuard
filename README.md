# ThreatSensor 🔍
> Real-time transaction anomaly detection pipeline using statistical methods, ensemble ML, and ARIMA time-series forecasting — built to mirror Amazon's Denied Party Screening infrastructure at scale.

---

## Overview

ThreatSensor screens 50,000 synthetic financial transactions to detect 500 injected anomalies across a 4-layer detection pipeline. Each layer adds sophistication: starting from simple statistical rules, progressing through unsupervised ML, supervised ML, and ending with an ensemble that combines all signals — exactly the architecture used in production financial screening systems.

**Anomaly rate: 1% (500 anomalies in 50,000 transactions)** — a realistic and challenging class imbalance that forces proper ML practice.

---

## Architecture

```
Raw Transactions (50,000)
        │
        ▼
┌─────────────────────────┐
│  Layer 1: Statistical   │  Z-score, IQR, multi-rule thresholding
│  (no training needed)   │  <1ms per transaction
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Layer 2: Isolation     │  Unsupervised — no labels required
│  Forest                 │  Detects sparse regions in feature space
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Layer 3: Supervised    │  Random Forest + XGBoost
│  ML (RF + XGBoost)      │  SMOTE oversampling, class weighting
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Layer 4: Time Series   │  ARIMA on hourly transaction volume
│  (ARIMA)                │  Flags abnormal spikes/drops
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Ensemble               │  Weighted combination of all scores
│  (Final Decision)       │  Optimized for PR-AUC
└─────────────────────────┘
```

---

## Project Structure

```
threatsensor/
├── main.py                   ← Entry point — runs the full pipeline
├── data_generator.py         ← Generates 50,000 synthetic transactions
├── eda.py                    ← Exploratory Data Analysis + 6-panel charts
├── statistical_detectors.py  ← Z-score, IQR, multi-rule detectors
├── ml_models.py              ← Isolation Forest, Random Forest, XGBoost, Ensemble
├── time_series.py            ← ARIMA forecasting + volume anomaly detection
├── requirements.txt
└── outputs/                  ← All charts saved here (auto-created on run)
```

---

## Concepts Implemented

### Statistics
| Concept | Where Used |
|---|---|
| Log-normal distribution | Transaction amounts — why raw amounts are skewed |
| Z-score `z = (x-μ)/σ` | `statistical_detectors.py` — flags >2.5σ deviations |
| IQR outlier detection | `statistical_detectors.py` — non-parametric fencing |
| Descriptive statistics | `eda.py` — mean, median, std, Q1/Q3 per class |
| Correlation analysis | `eda.py` — heatmap of feature-label correlations |

### Machine Learning
| Concept | Where Used |
|---|---|
| Isolation Forest | `ml_models.py` — unsupervised, path-length based scoring |
| Random Forest | `ml_models.py` — bagging + feature randomization |
| XGBoost | `ml_models.py` — sequential gradient boosting |
| SMOTE | `ml_models.py` — synthetic minority oversampling |
| Class weighting | `ml_models.py` — `class_weight='balanced'` |
| Feature importance | `ml_models.py` — Mean Decrease in Impurity |
| Threshold tuning | `ml_models.py` — 0.30 instead of default 0.50 |
| Ensemble scoring | `ml_models.py` — weighted average of normalized scores |

### Time Series
| Concept | Where Used |
|---|---|
| ADF stationarity test | `time_series.py` — determines differencing order `d` |
| ACF / PACF plots | `time_series.py` — identifies AR order `p` and MA order `q` |
| ARIMA(p,d,q) | `time_series.py` — fits on hourly transaction counts |
| Confidence intervals | `time_series.py` — 95% CI; values outside = volume anomaly |

### Evaluation
| Metric | Why It Matters |
|---|---|
| **PR-AUC** | Primary metric — accuracy is misleading at 99:1 class ratio |
| AUROC | Ranking quality across all thresholds |
| Precision | Of all flags raised, how many are real anomalies? |
| Recall | Of all real anomalies, how many did we catch? |
| F1 Score | Harmonic mean — single number balancing P and R |
| Confusion Matrix | Visual breakdown of TP / FP / TN / FN |

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Individual modules can also be run standalone:
```bash
python data_generator.py
python eda.py
python statistical_detectors.py
python ml_models.py
python time_series.py
```

---

## Expected Results

```
Model                  PR-AUC    AUROC    Recall    Precision
Isolation Forest       ~0.35     ~0.82    ~0.65     ~0.15
Random Forest          ~0.68     ~0.93    ~0.80     ~0.55
XGBoost                ~0.72     ~0.94    ~0.82     ~0.58
Ensemble               ~0.75     ~0.95    ~0.84     ~0.60
```

A random classifier scores PR-AUC = 0.01 (class prevalence). The ensemble at ~0.75 is **75x better than random**.

---

## Output Files

| File | What It Shows |
|---|---|
| `outputs/eda_plots.png` | 6-panel EDA: raw vs log-transformed amounts, hour distributions, boxplots, correlation heatmap |
| `outputs/statistical_confusion_matrices.png` | Side-by-side confusion matrices for all 3 statistical detectors |
| `outputs/feature_importance.png` | Which features drive Random Forest anomaly detection |
| `outputs/all_model_curves.png` | ROC + Precision-Recall curves for all models overlaid |
| `outputs/arima_diagnostics.png` | ACF/PACF plots + stationarity visualization |
| `outputs/arima_forecast.png` | 48-hour ARIMA forecast with 95% CI and flagged anomalies |

---

## Key Design Decisions

**Why log-transform amounts?**
Transaction amounts follow a log-normal distribution — always positive with a long right tail. Applying `log(x+1)` makes the distribution approximately normal, which is required for Z-score to be statistically valid and improves all distance-based models.

**Why PR-AUC over accuracy?**
With 99% normal transactions, a model predicting "normal" for everything achieves 99% accuracy but catches zero anomalies. PR-AUC measures performance on the minority class directly and is the correct metric for imbalanced screening problems.

**Why SMOTE and class_weight together?**
SMOTE generates synthetic anomaly examples in the training set. `class_weight='balanced'` additionally upweights the loss on anomaly predictions during training. Using both (belt and suspenders) provides the strongest possible signal about the rare class.

**Why Isolation Forest as an unsupervised layer?**
In production you don't always have labeled data for new attack patterns. Isolation Forest provides a strong anomaly signal purely from data structure, making it useful for detecting novel threats that weren't present in the training labels.
