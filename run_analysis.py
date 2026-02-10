"""
═══════════════════════════════════════════════════════════════
  Advanced Fraud Detection System
  A Hybrid ML Architecture for Regional Banking
═══════════════════════════════════════════════════════════════

  Author:  Justin Arndt
  Date:    February 2026
  Target:  Fulton Bank Data Scientist Role

  This script runs the complete fraud detection pipeline:
    1. Synthetic data generation (300K transactions, 3 attack vectors)
    2. Feature engineering (18 features: temporal, velocity, graph)
    3. Model training (PyTorch MLP + XGBoost baseline)
    4. Evaluation & benchmarking
    5. SHAP explainability (4 plot types, SR 11-7 compliant)
    6. Hero visualizations (10 publication-quality plots)

  Run:
    python run_analysis.py

  All outputs saved to: outputs/
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import FinancialDataSimulator
from src.feature_engineering import engineer_features, get_feature_descriptions
from src.model import (
    FraudMLP, prepare_data, train_mlp, 
    evaluate_model, train_xgboost_baseline, compare_models
)
from src.explainability import generate_all_shap_plots
from src.visualization import generate_all_hero_plots


def main():
    """Run the complete fraud detection analysis pipeline."""
    
    start_time = time.time()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("═" * 70)
    print("  ADVANCED FRAUD DETECTION SYSTEM")
    print("  A Hybrid ML Architecture for Regional Banking")
    print("═" * 70)
    
    # ─────────────────────────────────────────────────────
    # PHASE 1: DATA GENERATION
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 1: Synthetic Data Generation")
    print("─" * 70)
    
    simulator = FinancialDataSimulator(n_rows=300_000, fraud_rate=0.015, seed=42)
    df_raw = simulator.generate()
    
    # ─────────────────────────────────────────────────────
    # PHASE 2: FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 2: Feature Engineering")
    print("─" * 70)
    
    df_enriched, scaler, label_encoders, feature_names, num_cols, cat_cols = engineer_features(df_raw)
    
    # Print feature descriptions
    descriptions = get_feature_descriptions()
    print("\nEngineered Features:")
    for feat in feature_names[:len(descriptions)]:
        if feat in descriptions:
            print(f"  • {feat}: {descriptions[feat]}")
    
    # ─────────────────────────────────────────────────────
    # PHASE 3: MODEL TRAINING
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 3: Model Training")
    print("─" * 70)
    
    # Prepare data splits
    data = prepare_data(df_enriched, feature_names, batch_size=256)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Using device: {device}")
    
    # Train MLP
    model = FraudMLP(len(feature_names), dropout_rate=0.3).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_mlp(model, data, device, epochs=30, lr=1e-3, patience=5)
    
    # ─────────────────────────────────────────────────────
    # PHASE 4: EVALUATION
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 4: Model Evaluation")
    print("─" * 70)
    
    mlp_results = evaluate_model(model, data, device, model_name="Neural Network (MLP)")
    
    # XGBoost baseline
    xgb_model, xgb_results = train_xgboost_baseline(data, feature_names)
    
    # Side-by-side comparison
    compare_models(mlp_results, xgb_results)
    
    # ─────────────────────────────────────────────────────
    # PHASE 5: SHAP EXPLAINABILITY
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 5: SHAP Explainability (SR 11-7)")
    print("─" * 70)
    
    shap_values, expected_value, X_explain_df = generate_all_shap_plots(
        model, data['X_train'], data['X_test'], data['y_test'],
        feature_names, output_dir=output_dir, n_explain=200, device_str='cpu'
    )
    
    # Move model back to device after SHAP (which runs on CPU)
    model.to(device)
    
    # ─────────────────────────────────────────────────────
    # PHASE 6: HERO VISUALIZATIONS
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 6: Hero Visualizations")
    print("─" * 70)
    
    generate_all_hero_plots(
        df_raw, mlp_results, xgb_results, history,
        model, data, device,
        shap_values=shap_values,
        feature_names=feature_names,
        xgb_importance=xgb_results.get('feature_importance'),
        output_dir=output_dir
    )
    
    # ─────────────────────────────────────────────────────
    # PHASE 7: REAL-TIME INFERENCE BENCHMARK
    # ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 7: Real-Time Inference Benchmark")
    print("─" * 70)
    
    model.eval()
    n_benchmark = 100
    latencies = []
    with torch.no_grad():
        for i in range(n_benchmark):
            sample = torch.from_numpy(
                data['X_test'].iloc[i:i+1].values.astype(np.float32)
            ).float().to(device)
            t0 = time.time()
            _ = torch.sigmoid(model(sample))
            latencies.append((time.time() - t0) * 1000)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"  Average inference latency: {avg_latency:.2f} ms")
    print(f"  P95 inference latency:     {p95_latency:.2f} ms")
    print(f"  FedNow compatible (<100ms): {'✓ YES' if p95_latency < 100 else '✗ NO'}")
    
    # ─────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)
    print(f"  Total runtime:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Transactions:      {len(df_raw):,}")
    print(f"  Features:          {len(feature_names)}")
    print(f"  MLP ROC-AUC:       {mlp_results['roc_auc']:.4f}")
    print(f"  MLP PR-AUC:        {mlp_results['pr_auc']:.4f}")
    print(f"  XGB ROC-AUC:       {xgb_results['roc_auc']:.4f}")
    print(f"  XGB PR-AUC:        {xgb_results['pr_auc']:.4f}")
    print(f"  Inference latency: {avg_latency:.2f}ms (P95: {p95_latency:.2f}ms)")
    print(f"  Outputs saved to:  {os.path.abspath(output_dir)}/")
    print("═" * 70)
    
    return {
        'model': model,
        'xgb_model': xgb_model,
        'mlp_results': mlp_results,
        'xgb_results': xgb_results,
        'history': history,
        'data': data,
        'df_raw': df_raw,
        'feature_names': feature_names,
        'shap_values': shap_values,
    }


if __name__ == "__main__":
    results = main()
