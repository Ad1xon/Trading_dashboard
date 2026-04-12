import pandas as pd
import numpy as np
from quant_engine.ml_models.xgb_model import XGBoostRangeBarModel

np.random.seed(42)
n = 1000
dates = pd.date_range('2023-01-01', periods=n, freq='15T')
close = np.random.normal(0, 0.001, n).cumsum() + 1.1000
open_p = close + np.random.normal(0, 0.0005, n)
high_p = np.maximum(close, open_p) + np.abs(np.random.normal(0, 0.0005, n))
low_p = np.minimum(close, open_p) - np.abs(np.random.normal(0, 0.0005, n))
volume = np.abs(np.random.normal(100, 50, n))

df = pd.DataFrame({'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close, 'Volume': volume}, index=dates)

model = XGBoostRangeBarModel()
features_df = model.build_features(df)
feat_cols = model.FEATURE_COLS

corr_matrix = features_df[feat_cols].corr()

print("Highly correlated features (> 0.75 or < -0.75):")
for i in range(len(feat_cols)):
    for j in range(i+1, len(feat_cols)):
        col1 = feat_cols[i]
        col2 = feat_cols[j]
        if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
            val = corr_matrix.loc[col1, col2]
            if abs(val) > 0.75:
                print(f"{col1} - {col2}: {val:.2f}")

print("\nAll correlations analyzed.")
