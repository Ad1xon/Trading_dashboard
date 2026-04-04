import pandas as pd
from ..indicators import calculate_vwap_with_bands
from ..stat_arb import test_cointegration
from ..ml_models import StatArbMLFilter

def detect_liquidity_sweep(df: pd.DataFrame) -> dict:
    df = calculate_vwap_with_bands(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['Low'] < prev['VWAP_Lower_2'] and latest['Close'] > latest['VWAP_Lower_2']:
        return {"signal": True, "type": "BULLISH_SWEEP",
                "message": f"Byczy Liquidity Sweep: {latest['VWAP_Lower_2']:.2f}."}
    elif prev['High'] > prev['VWAP_Upper_2'] and latest['Close'] < latest['VWAP_Upper_2']:
        return {"signal": True, "type": "BEARISH_SWEEP",
                "message": f"Niedźwiedzi Liquidity Sweep: {latest['VWAP_Upper_2']:.2f}."}
    return {"signal": False, "type": None, "message": ""}

def analyze_pair_opportunity(df_y: pd.DataFrame, df_x: pd.DataFrame, ml_filter: StatArbMLFilter) -> dict:
    arb_data = test_cointegration(df_y['Close'], df_x['Close'])
    if not arb_data["is_cointegrated"]:
        return {"signal": False, "message": "Brak kointegracji statystycznej."}
    latest_z = arb_data["z_score"].iloc[-1]
    if abs(latest_z) >= 2.0:
        features = ml_filter.prepare_features(arb_data["spread"], arb_data["z_score"])
        if features.empty:
            return {"signal": False, "message": "Brak danych ML."}
        latest_row = features.drop('Target', axis=1).iloc[-1:]
        prob_success = ml_filter.predict_probability(latest_row)
        if prob_success > 0.65:
            action = "SHORT Y, LONG X" if latest_z > 0 else "LONG Y, SHORT X"
            return {"signal": True,
                    "message": f"Setup StatArb. Z-Score: {latest_z:.2f}. ML szanse: {prob_success * 100:.1f}%. {action}"}
        return {"signal": False, "message": "Odrzucono przez model ML."}
    return {"signal": False, "message": "Z-score w normie."}