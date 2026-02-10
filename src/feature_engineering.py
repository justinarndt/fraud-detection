"""
Feature Engineering Pipeline
==============================
Transforms raw transaction data into ML-ready feature vectors:
  - Cyclic temporal encoding (sin/cos for hour and day-of-week)
  - Velocity aggregations (transaction frequency, amount deviation)
  - Graph topology features (device centrality)
  - Categorical encoding for model consumption

Designed for SR 11-7 compliance: every transformation is documented and reproducible.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df, fit_scalers=True, scaler=None, label_encoders=None):
    """
    Full feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data from FinancialDataSimulator
    fit_scalers : bool
        If True, fit new scaler/encoders. If False, use provided ones (for inference).
    scaler : StandardScaler or None
        Pre-fitted scaler for inference mode.
    label_encoders : dict or None
        Pre-fitted label encoders for inference mode.
        
    Returns
    -------
    tuple: (df_engineered, scaler, label_encoders, feature_names, num_cols, cat_cols)
    """
    print("[Features] Starting feature engineering pipeline...")
    df = df.copy()
    
    # ─────────────────────────────────────────────────────
    # 1. CYCLIC TEMPORAL ENCODING
    # Neural networks struggle with hour 23 → hour 0 discontinuity.
    # Mapping to unit circle preserves temporal proximity.
    # ─────────────────────────────────────────────────────
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    print("  → Cyclic time encoding: hour_sin, hour_cos, dow_sin, dow_cos")
    
    # ─────────────────────────────────────────────────────
    # 2. VELOCITY & BEHAVIORAL AGGREGATIONS
    # Detect bust-out attacks by measuring deviation from baseline.
    # In production: computed in real-time via Redis/Timestream.
    # ─────────────────────────────────────────────────────
    df['cust_txn_count'] = df.groupby('user_id')['transaction_id'].transform('count')
    df['cust_avg_amount'] = df.groupby('user_id')['amount'].transform('mean')
    df['cust_std_amount'] = df.groupby('user_id')['amount'].transform('std').fillna(0)
    df['amount_zscore'] = (df['amount'] - df['cust_avg_amount']) / (df['cust_std_amount'] + 1e-8)
    df['amount_ratio'] = df['amount'] / (df['cust_avg_amount'] + 1e-5)
    
    # Log-transform amount to reduce skew
    df['log_amount'] = np.log1p(df['amount'])
    print("  → Velocity features: cust_txn_count, amount_zscore, amount_ratio, log_amount")
    
    # ─────────────────────────────────────────────────────
    # 3. GRAPH TOPOLOGY FEATURES
    # Device degree already computed in data_generator.
    # Add user-level graph metrics.
    # In production: computed via Reltio entity resolution.
    # ─────────────────────────────────────────────────────
    df['user_device_count'] = df.groupby('user_id')['device_id'].transform('nunique')
    df['device_user_count'] = df.groupby('device_id')['user_id'].transform('nunique')
    print("  → Graph features: device_degree, user_device_count, device_user_count")
    
    # ─────────────────────────────────────────────────────
    # 4. INTERACTION FEATURES
    # Cross-feature signals that amplify fraud detection.
    # ─────────────────────────────────────────────────────
    df['velocity_x_amount'] = df['velocity_24h'] * df['log_amount']
    df['night_high_amount'] = ((df['hour'] < 5) | (df['hour'] > 22)).astype(int) * df['log_amount']
    print("  → Interaction features: velocity_x_amount, night_high_amount")
    
    # ─────────────────────────────────────────────────────
    # 5. NUMERICAL SCALING
    # StandardScaler for neural network convergence.
    # ─────────────────────────────────────────────────────
    num_cols = [
        'amount', 'log_amount', 'velocity_24h', 'device_degree',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'device_shared',
        'cust_txn_count', 'cust_avg_amount', 'amount_zscore', 'amount_ratio',
        'user_device_count', 'device_user_count',
        'velocity_x_amount', 'night_high_amount'
    ]
    
    if fit_scalers:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    print(f"  → Scaled {len(num_cols)} numerical features")
    
    # ─────────────────────────────────────────────────────
    # 6. CATEGORICAL ENCODING
    # One-hot for low-cardinality categoricals.
    # ─────────────────────────────────────────────────────
    cat_cols_raw = ['card_type', 'entry_mode']
    
    if fit_scalers:
        label_encoders = {}
        for col in cat_cols_raw:
            le = LabelEncoder()
            df[f'{col}_idx'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols_raw:
            df[f'{col}_idx'] = label_encoders[col].transform(df[col].astype(str))
    
    # Also create one-hot for interpretable models (XGBoost)
    cat_dummies = pd.get_dummies(df[cat_cols_raw], prefix=['card', 'entry'])
    cat_cols = list(cat_dummies.columns)
    df = pd.concat([df, cat_dummies], axis=1)
    print(f"  → Encoded {len(cat_cols_raw)} categorical features → {len(cat_cols)} one-hot columns")
    
    # ─────────────────────────────────────────────────────
    # 7. FINAL FEATURE SET
    # ─────────────────────────────────────────────────────
    feature_names = num_cols + cat_cols
    
    print(f"\n[Features] Complete. {len(feature_names)} features engineered.")
    print(f"[Features] Numerical: {len(num_cols)} | Categorical (one-hot): {len(cat_cols)}")
    
    return df, scaler, label_encoders, feature_names, num_cols, cat_cols


def get_feature_descriptions():
    """
    Returns a dictionary mapping feature names to human-readable descriptions.
    Used for SHAP interpretation and SR 11-7 documentation.
    """
    return {
        'amount': 'Transaction amount (USD, scaled)',
        'log_amount': 'Log-transformed transaction amount',
        'velocity_24h': 'Transactions in last 24 hours',
        'device_degree': 'Number of unique users on this device (graph centrality)',
        'hour_sin': 'Hour of day (sine component, cyclic encoding)',
        'hour_cos': 'Hour of day (cosine component, cyclic encoding)',
        'dow_sin': 'Day of week (sine component, cyclic encoding)',
        'dow_cos': 'Day of week (cosine component, cyclic encoding)',
        'is_weekend': 'Weekend indicator (1 = Sat/Sun)',
        'device_shared': 'Device sharing indicator (1 = shared)',
        'cust_txn_count': 'Total transactions by this customer',
        'cust_avg_amount': 'Customer average transaction amount (scaled)',
        'amount_zscore': 'Amount z-score relative to customer history',
        'amount_ratio': 'Amount / customer average amount',
        'user_device_count': 'Number of devices used by this customer',
        'device_user_count': 'Number of users on this device',
        'velocity_x_amount': 'Interaction: velocity × log amount',
        'night_high_amount': 'Interaction: nighttime × log amount',
    }
