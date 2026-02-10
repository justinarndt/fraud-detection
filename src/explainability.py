"""
SHAP Explainability Module
============================
Reliable SHAP rendering for SR 11-7 compliance:
  - Global summary (bar plot)
  - Global beeswarm (impact direction)
  - Local waterfall (single-instance explanation)
  - Local force plot (matplotlib-rendered)

Key design decisions:
  - Uses DeepExplainer on CPU with small background set (200 samples)
  - All plots rendered via matplotlib (not JavaScript) for guaranteed output
  - Saves all figures to disk as publication-quality PNGs
"""

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for reliable rendering
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def compute_shap_values(model, X_train, X_explain, feature_names, 
                        n_background=200, device='cpu'):
    """
    Compute SHAP values using DeepExplainer.
    
    Parameters
    ----------
    model : FraudMLP
        Trained PyTorch model
    X_train : pd.DataFrame
        Training data for background distribution
    X_explain : pd.DataFrame
        Data instances to explain
    feature_names : list
        Feature column names
    n_background : int
        Number of background samples (keep small for memory)
    device : str
        Device to compute on (use 'cpu' for SHAP compatibility)
        
    Returns
    -------
    tuple: (shap_values, expected_value, X_explain_df)
    """
    print("[SHAP] Computing SHAP values...")
    print(f"  → Background samples: {n_background}")
    print(f"  → Instances to explain: {len(X_explain)}")
    
    # Move model to CPU for SHAP compatibility
    model_cpu = model.cpu()
    model_cpu.eval()
    
    # Prepare background data
    bg_indices = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
    background = torch.from_numpy(
        X_train.iloc[bg_indices].values.astype(np.float32)
    ).float()
    
    # Prepare explanation data
    explain_tensor = torch.from_numpy(
        X_explain.values.astype(np.float32)
    ).float()
    
    # Compute SHAP values
    explainer = shap.DeepExplainer(model_cpu, background)
    shap_values = explainer.shap_values(explain_tensor)
    
    # Handle output format
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.numpy()
    
    # Flatten if needed (remove output dimension)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    elif shap_values.ndim == 2 and shap_values.shape[1] != len(feature_names):
        shap_values = shap_values.squeeze()
    
    # Get expected value
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.item()
    if isinstance(expected_value, torch.Tensor):
        expected_value = expected_value.item()
    
    # Prepare DataFrame for display
    X_explain_df = X_explain.copy()
    X_explain_df.columns = feature_names
    
    print(f"  → SHAP values shape: {shap_values.shape}")
    print(f"  → Expected value (base): {expected_value:.4f}")
    print("[SHAP] Complete.")
    
    return shap_values, expected_value, X_explain_df


def plot_shap_summary_bar(shap_values, X_explain_df, save_path=None):
    """
    SHAP Global Feature Importance (Bar Plot).
    Shows mean |SHAP value| per feature — which features matter most overall.
    """
    print("[SHAP] Generating global importance bar plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, X_explain_df,
        plot_type="bar",
        show=False,
        max_display=20
    )
    
    plt.title("Global Feature Importance (SHAP)\nMean |SHAP value| — Which Features Drive Fraud Decisions",
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Mean |SHAP Value|", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  → Saved to {save_path}")
    
    plt.show()
    return fig


def plot_shap_beeswarm(shap_values, X_explain_df, save_path=None):
    """
    SHAP Beeswarm Plot.
    Shows feature value impact direction — red = high value, blue = low value.
    """
    print("[SHAP] Generating beeswarm plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, X_explain_df,
        plot_type="dot",
        show=False,
        max_display=20
    )
    
    plt.title("Feature Impact Direction (SHAP Beeswarm)\nRed = High Feature Value → Positive Impact on Fraud Score",
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("SHAP Value (Impact on Fraud Probability)", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  → Saved to {save_path}")
    
    plt.show()
    return fig


def plot_shap_waterfall(shap_values, expected_value, X_explain_df, 
                        instance_idx=0, save_path=None):
    """
    SHAP Waterfall Plot for a single instance.
    Shows exactly WHY this specific transaction was flagged/cleared.
    Critical for SR 11-7 local explainability.
    """
    print(f"[SHAP] Generating waterfall plot for instance {instance_idx}...")
    
    sv = shap_values[instance_idx]
    if sv.ndim > 1:
        sv = sv.squeeze()
    
    explanation = shap.Explanation(
        values=sv,
        base_values=float(expected_value),
        data=X_explain_df.iloc[instance_idx].values,
        feature_names=X_explain_df.columns.tolist()
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    
    plt.title(f"Transaction #{instance_idx} — Why Was This Flagged?\nSR 11-7 Local Explainability",
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  → Saved to {save_path}")
    
    plt.show()
    return fig


def plot_shap_force(shap_values, expected_value, X_explain_df,
                    instance_idx=0, save_path=None):
    """
    SHAP Force Plot for a single instance (matplotlib rendering).
    Compact horizontal view of feature contributions.
    """
    print(f"[SHAP] Generating force plot for instance {instance_idx}...")
    
    sv = shap_values[instance_idx]
    if sv.ndim > 1:
        sv = sv.squeeze()
    
    fig = plt.figure(figsize=(14, 3))
    shap.force_plot(
        float(expected_value),
        sv,
        X_explain_df.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    
    plt.title(f"Force Plot — Transaction #{instance_idx} Feature Contributions",
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  → Saved to {save_path}")
    
    plt.show()
    return fig


def generate_all_shap_plots(model, X_train, X_test, y_test, feature_names,
                            output_dir="outputs", n_explain=200, device_str='cpu'):
    """
    Generate all 4 SHAP plot types and save to disk.
    
    This is the main entry point for explainability reporting.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute SHAP values
    shap_values, expected_value, X_explain_df = compute_shap_values(
        model, X_train, X_test.iloc[:n_explain], feature_names,
        n_background=200, device=device_str
    )
    
    # 1. Global bar plot
    plot_shap_summary_bar(
        shap_values, X_explain_df,
        save_path=os.path.join(output_dir, "shap_global_importance.png")
    )
    
    # 2. Beeswarm
    plot_shap_beeswarm(
        shap_values, X_explain_df,
        save_path=os.path.join(output_dir, "shap_beeswarm.png")
    )
    
    # 3. Find a fraud instance for local plots
    y_subset = y_test.iloc[:n_explain]
    fraud_indices = np.where(y_subset.values == 1)[0]
    
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        print(f"\n[SHAP] Using fraud instance at index {fraud_idx} for local plots")
        
        # 4. Waterfall
        plot_shap_waterfall(
            shap_values, expected_value, X_explain_df,
            instance_idx=fraud_idx,
            save_path=os.path.join(output_dir, "shap_waterfall_fraud.png")
        )
        
        # 5. Force plot
        plot_shap_force(
            shap_values, expected_value, X_explain_df,
            instance_idx=fraud_idx,
            save_path=os.path.join(output_dir, "shap_force_fraud.png")
        )
    else:
        print("[SHAP] Warning: No fraud instances in explanation subset")
    
    # Also do a legit instance for comparison
    legit_indices = np.where(y_subset.values == 0)[0]
    if len(legit_indices) > 0:
        legit_idx = legit_indices[0]
        plot_shap_waterfall(
            shap_values, expected_value, X_explain_df,
            instance_idx=legit_idx,
            save_path=os.path.join(output_dir, "shap_waterfall_legit.png")
        )
    
    print(f"\n[SHAP] All plots saved to {output_dir}/")
    return shap_values, expected_value, X_explain_df
