"""
Synthetic Financial Data Generator
===================================
Generates realistic banking transaction data with injected fraud patterns:
  - Device Farm attacks (graph anomaly)
  - Typosquatting/phishing (NLP anomaly)  
  - Velocity bust-out (temporal anomaly)

Designed to simulate Fulton Bank-scale transaction data for ML training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class FinancialDataSimulator:
    """
    Generates synthetic financial transaction data with realistic fraud patterns.
    
    Parameters
    ----------
    n_rows : int
        Number of transactions to generate (default: 300,000)
    fraud_rate : float
        Base fraud injection rate (default: 0.015 = 1.5%)
    seed : int
        Random seed for reproducibility (SR 11-7 compliance)
    """
    
    def __init__(self, n_rows=300_000, fraud_rate=0.015, seed=42):
        self.n_rows = n_rows
        self.fraud_rate = fraud_rate
        self.seed = seed
        
    def generate(self):
        """
        Generate the full synthetic dataset.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: user_id, device_id, hour, day_of_week, 
            is_weekend, amount, velocity_24h, merchant, device_shared, is_fraud
        """
        np.random.seed(self.seed)
        print(f"[DataGen] Generating {self.n_rows:,} synthetic transactions...")
        
        # --- Entity backbone ---
        n_users = max(1000, self.n_rows // 6)
        n_devices = max(800, self.n_rows // 8)
        user_ids = np.arange(n_users)
        device_pool = np.arange(n_devices)
        
        # --- Temporal backbone ---
        # Realistic hour distribution: low at night, peak during business hours
        hour_weights = np.array([
            0.01, 0.005, 0.003, 0.003, 0.005, 0.01,   # 0-5 AM (very low)
            0.03, 0.06, 0.08, 0.09, 0.09, 0.08,        # 6-11 AM (rising)
            0.07, 0.06, 0.05, 0.05, 0.06, 0.07,        # 12-5 PM (steady)
            0.06, 0.05, 0.04, 0.03, 0.02, 0.015        # 6-11 PM (declining)
        ])
        hour_weights /= hour_weights.sum()
        hours = np.random.choice(24, self.n_rows, p=hour_weights)
        day_of_week = np.random.choice(7, self.n_rows, p=[0.16, 0.16, 0.16, 0.16, 0.16, 0.10, 0.10])
        is_weekend = (day_of_week >= 5).astype(int)
        
        # --- Transaction amounts (log-normal, realistic banking distribution) ---
        amounts = np.random.lognormal(mean=3.8, sigma=1.3, size=self.n_rows)
        amounts = np.clip(np.round(amounts, 2), 0.50, 15000.00)
        
        # --- Velocity (transactions in last 24h per user) ---
        velocity_24h = np.random.poisson(4, self.n_rows)
        
        # --- Merchant names with realistic variation ---
        base_merchants = [
            'Walmart', 'Amazon', 'Target', 'Costco', 'Uber', 'Lyft',
            'Shell Gas', 'Wawa', 'Sheetz', 'Starbucks', 'McDonalds',
            'Home Depot', 'CVS Pharmacy', 'Walgreens', 'Netflix',
            'Apple Store', 'Best Buy', 'Chipotle', 'Dunkin', 'Venmo P2P'
        ]
        merchant_suffixes = ['', ' #1234', ' Store', ' Online', ' *Purchase', '']
        
        raw_merchants = np.random.choice(base_merchants, self.n_rows)
        raw_suffixes = np.random.choice(merchant_suffixes, self.n_rows)
        merchants = np.array([f"{m}{s}" for m, s in zip(raw_merchants, raw_suffixes)])
        
        # Add realistic typos to ~8% of merchant names
        typo_mask = np.random.rand(self.n_rows) < 0.08
        for idx in np.where(typo_mask)[0]:
            name = merchants[idx]
            if 'o' in name.lower():
                merchants[idx] = name.replace('o', '0', 1)
            elif 'a' in name.lower():
                merchants[idx] = name.replace('a', '@', 1)
        
        # --- Entity assignment ---
        user_col = np.random.choice(user_ids, self.n_rows)
        device_col = np.random.choice(device_pool, self.n_rows)
        device_shared = np.random.choice([0, 1], self.n_rows, p=[0.95, 0.05])
        
        # --- Card type & entry mode ---
        card_types = np.random.choice(
            ['Visa', 'Mastercard', 'Amex', 'Discover'],
            self.n_rows, p=[0.50, 0.30, 0.15, 0.05]
        )
        entry_modes = np.random.choice(
            ['Chip', 'Contactless', 'Swipe', 'Manual', 'E-Commerce'],
            self.n_rows, p=[0.30, 0.25, 0.15, 0.05, 0.25]
        )
        
        # --- Initialize labels ---
        labels = np.zeros(self.n_rows, dtype=int)
        n_fraud = int(self.n_rows * self.fraud_rate)
        
        # ═══════════════════════════════════════════════════
        # ATTACK VECTOR A: DEVICE FARM (Graph Anomaly)
        # One device shared across many unrelated users
        # Simulates synthetic identity ring
        # ═══════════════════════════════════════════════════
        n_farm = n_fraud // 3
        farm_indices = np.random.choice(self.n_rows, n_farm, replace=False)
        farm_device_id = n_devices + 1  # Unique fraud device
        device_col[farm_indices] = farm_device_id
        device_shared[farm_indices] = 1
        labels[farm_indices] = 1
        amounts[farm_indices] *= np.random.uniform(1.2, 2.5, n_farm)
        print(f"  → Injected {n_farm:,} device farm fraud transactions")
        
        # ═══════════════════════════════════════════════════
        # ATTACK VECTOR B: TYPOSQUATTING (NLP Anomaly)
        # Fake merchant names mimicking real ones
        # ═══════════════════════════════════════════════════
        n_phish = n_fraud // 3
        remaining = np.where(labels == 0)[0]
        phish_indices = np.random.choice(remaining, n_phish, replace=False)
        phish_merchants = ['Amaz0n Marketplace', 'PayPaI Services', 'App1e Store',
                          'Wa1mart Online', 'Netf1ix Billing', 'G00gle Play']
        merchants[phish_indices] = np.random.choice(phish_merchants, n_phish)
        entry_modes[phish_indices] = 'E-Commerce'
        labels[phish_indices] = 1
        print(f"  → Injected {n_phish:,} typosquatting fraud transactions")
        
        # ═══════════════════════════════════════════════════
        # ATTACK VECTOR C: VELOCITY BUST-OUT (Temporal Anomaly)
        # Sudden burst of high-value transactions
        # ═══════════════════════════════════════════════════
        n_bust = n_fraud - n_farm - n_phish
        remaining = np.where(labels == 0)[0]
        bust_indices = np.random.choice(remaining, n_bust, replace=False)
        amounts[bust_indices] = amounts[bust_indices] * np.random.uniform(5, 15, n_bust) + 500
        velocity_24h[bust_indices] += np.random.poisson(25, n_bust)
        hours[bust_indices] = np.random.choice([0, 1, 2, 3, 4, 23], n_bust)  # Late night
        labels[bust_indices] = 1
        print(f"  → Injected {n_bust:,} velocity bust-out fraud transactions")
        
        # --- Assemble DataFrame ---
        df = pd.DataFrame({
            'transaction_id': np.arange(self.n_rows),
            'user_id': user_col,
            'device_id': device_col,
            'hour': hours,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'amount': np.round(amounts, 2),
            'velocity_24h': velocity_24h,
            'merchant': merchants,
            'card_type': card_types,
            'entry_mode': entry_modes,
            'device_shared': device_shared,
            'is_fraud': labels
        })
        
        # --- Graph feature: device degree (number of unique users per device) ---
        device_user_counts = df.groupby('device_id')['user_id'].nunique()
        df['device_degree'] = df['device_id'].map(device_user_counts)
        
        actual_fraud_rate = df['is_fraud'].mean()
        print(f"\n[DataGen] Complete. Shape: {df.shape}")
        print(f"[DataGen] Fraud rate: {actual_fraud_rate:.4%} ({df['is_fraud'].sum():,} fraud / {len(df):,} total)")
        print(f"[DataGen] Attack vectors: Device Farm + Typosquatting + Bust-Out")
        
        return df


if __name__ == "__main__":
    sim = FinancialDataSimulator(n_rows=10_000)
    df = sim.generate()
    print(f"\nSample:\n{df.head()}")
    print(f"\nFraud distribution:\n{df['is_fraud'].value_counts()}")
