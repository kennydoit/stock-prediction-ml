# Stock Price Prediction with Machine Learning

This project implements a machine learning model to predict stock price movements for Microsoft (MSFT) using technical indicators and peer company analysis.

## Project Structure

```
stock-prediction-ml/
├── data/
│   ├── raw/                # Downloaded CSVs or API data dumps
│   ├── processed/          # Cleaned & engineered datasets
├── notebooks/
│   ├── exploratory.ipynb   # EDA and visualization
├── src/
│   ├── data_loader.py      # Pull data from API or local CSVs
│   ├── features.py         # RSI, MA, MACD, etc.
│   ├── model.py            # Training and evaluation logic
│   ├── predict.py          # Forecasting MSFT based on peers
├── models/
│   ├── xgboost_msft.model
├── requirements.txt
├── config.yaml             # Symbol list, timeframes, API keys
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure settings in `config.yaml`
2. Run the prediction script:
```bash
python src/predict.py
```

3. For exploratory analysis, launch Jupyter Notebook:
```bash
jupyter notebook notebooks/exploratory.ipynb
```

## Features

- Data loading from Yahoo Finance API
- Technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- XGBoost regression model
- Configurable prediction horizon
- Peer company analysis

## License

MIT License
