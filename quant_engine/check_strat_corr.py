import pandas as pd
import numpy as np
from quant_engine.strategies import STRATEGY_REGISTRY

np.random.seed(42)
n = 1000
dates = pd.date_range('2023-01-01', periods=n, freq='15T')
close = np.random.normal(0, 0.001, n).cumsum() + 1.1000
open_p = close + np.random.normal(0, 0.0005, n)
high_p = np.maximum(close, open_p) + np.abs(np.random.normal(0, 0.0005, n))
low_p = np.minimum(close, open_p) - np.abs(np.random.normal(0, 0.0005, n))
volume = np.abs(np.random.normal(100, 50, n))

df = pd.DataFrame({'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close, 'Volume': volume}, index=dates)

signals = {}
for name, cls in STRATEGY_REGISTRY.items():
    try:
        strat = cls()
        strat_df = df.copy()
        strat.generate_signals(strat_df)
        if 'Signal' in strat_df.columns:
            signals[name] = strat_df['Signal']
    except Exception as e:
        print(f"Error in strategy {name}: {e}")

if signals:
    sig_df = pd.DataFrame(signals)
    corr_matrix = sig_df.corr()
    print("Strategy Output Correlations:")
    print(corr_matrix)
    print("\nHighly correlated strategies (> 0.75 or < -0.75):")
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.75:
                print(f"{cols[i]} - {cols[j]}: {val:.2f}")
else:
    print("No signal arrays found.")
