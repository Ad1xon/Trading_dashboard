# Institutional Quant Engine & Dashboard

A comprehensive, production-ready quantitative trading engine and dashboard designed for rigorous backtesting, live market scanning, and algorithmic signal generation.

## Overview

This repository contains an advanced algorithmic trading framework that integrates machine learning prediction models, statistical mean-reversion, intraday volume scalping, and global macroeconomic sentiment analysis. The system is built for institutional-grade reliability, preventing look-ahead bias and incorporating exact transaction costs (slippage/spread).

### Key Features
- **Dynamic Market Scanner**: Real-time multi-timeframe scanner identifying liquidity sweeps, VWAP setups, and RSI extremes.
- **Advanced Backtester**: Fully vectorized and walk-forward optimized backtesting engine. Supports custom Range Bars and exact MetaTrader 5 tick-level simulation.
- **Machine Learning Integration**: Built-in XGBoost, LightGBM, and PyTorch LSTM models utilizing Purged Walk-Forward Cross-Validation.
- **Risk Management**: Dynamic ATR-based Stop Loss and Take Profit levels with account margin call circuit breakers and VaR/CVaR risk metrics.
- **Global Macro Filter**: FinBERT-powered NLP sentiment analysis on live news and High-Impact economic calendar events integration.
- **Alert System**: Centralized Event Bus architecture dispatching deduped trading signals directly to Discord webhooks.

## Architecture

The project is structured into logically segregated components for scalability:

- `quant_engine/`: Core algorithms.
  - `ml_models/`: Walk-forward trained ML classifiers (XGB, LGBM, PyTorch LSTM).
  - `strategies/`: Registry of available trading logics (VWAP Bounce, Arabian Scalper, SMC Breakouts).
  - `indicators.py`: Vectorized technical indicators (VWAP bands, RSI, ADX).
  - `scanner.py`: Background market scanning and signal compilation logic.
  - `backtester.py`: OOS strategy performance evaluation.
- `dashboard/`: Streamlit-based graphical user interface (`app.py`, `views.py`).
- `data_feed/`: MetaTrader 5 live integration and NLP RSS feeds.
- `alerts/`: Dedicated module for dispatching signals.

## Trading Strategies

All strategies inherit from `BaseStrategy` and dynamically generate entry/exit and SL/TP points:
1. **VWAP Bounce**: Mean-reversion trading extreme standard deviations from the daily anchored VWAP.
2. **Arabian Volume Scalper**: Momentum logic utilizing LightGBM probabilities and rolling volume surge confirmation.
3. **SMC Volatility Breakout**: Smart Money Concept-inspired breakouts factoring in volatility contraction.
4. **LSTM Swing**: PyTorch-based sequential neural network for multi-day directional holding.
5. **MultiTimeframe Momentum**: Slow-MA alignment with Fast-RSI retracements.

## Tech Stack

- **Data Processing**: Pandas, Numpy, Scikit-Learn
- **Machine Learning**: XGBoost, LightGBM, PyTorch
- **Natural Language Processing**: HuggingFace Transformers (FinBERT)
- **Broker Integration**: MetaTrader5 API
- **User Interface**: Streamlit, Plotly
- **Alerts**: Discord Webhooks via Async Event Bus

## Setup & Installation

**Prerequisites:**
- Python 3.10+
- MetaTrader 5 Terminal running locally and logged into an active account.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/quant-engine.git
cd quant-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Quant Dashboard
streamlit run dashboard/app.py
```

## Backtester Usage

Navigate to the **Advanced Backtester** tab in the dashboard.
1. Select your desired instrument (e.g., EURUSD, XAUUSD, NAS100).
2. Choose from the `STRATEGY_REGISTRY`.
3. Configure the backtest horizon, initial capital, and risk sizing.
4. Run the simulation to view Equity Curves, Trade Distributions, VaR profiles, and ML Feature Importance charts.

---
*Disclaimer: This software is for research and educational purposes only. Do not deploy algorithms on live accounts without rigorous paper-trading and capital risk assessment.*
