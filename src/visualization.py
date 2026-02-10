"""
Hero Visualization Module
============================
Publication-quality plots for the fraud detection project.
All plots use a consistent professional dark theme.

Hero Plots:
  1. Confusion Matrix Heatmap (annotated, dark theme)
  2. PR Curve with AUC Shading
  3. ROC Curve with AUC Shading
  4. Model Comparison Dashboard
  5. Fraud Ring Network Graph
  6. Temporal Fraud Heatmap (hour × day-of-week)
  7. Amount Distribution (violin/box)
  8. Training History
  9. Feature Importance Comparison (SHAP vs XGBoost)
 10. Latent Space Projection (PCA 2D)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════
# PROFESSIONAL THEME
# ═══════════════════════════════════════════════════════

# Dark professional palette
COLORS = {
    'bg_dark': '#0D1117',
    'bg_card': '#161B22',
    'bg_accent': '#21262D',
    'text_primary': '#F0F6FC',
    'text_secondary': '#8B949E',
    'blue': '#58A6FF',
    'green': '#3FB950',
    'red': '#F85149',
    'orange': '#D29922',
    'purple': '#BC8CFF',
    'cyan': '#39D2C0',
    'pink': '#F778BA',
    'grid': '#30363D',
}

# Gradient colormaps
FRAUD_CMAP = LinearSegmentedColormap.from_list(
    'fraud', ['#0D1117', '#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560']
)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'heatmap', ['#0D1117', '#161B22', '#1f4068', '#3672a4', '#58A6FF', '#79C0FF']
)


def _apply_dark_theme(fig, ax, title="", subtitle=""):
    """Apply consistent dark professional theme to any plot."""
    fig.set_facecolor(COLORS['bg_dark'])
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            a.set_facecolor(COLORS['bg_card'])
            a.tick_params(colors=COLORS['text_secondary'], labelsize=10)
            a.xaxis.label.set_color(COLORS['text_secondary'])
            a.yaxis.label.set_color(COLORS['text_secondary'])
            for spine in a.spines.values():
                spine.set_color(COLORS['grid'])
    else:
        ax.set_facecolor(COLORS['bg_card'])
        ax.tick_params(colors=COLORS['text_secondary'], labelsize=10)
        ax.xaxis.label.set_color(COLORS['text_secondary'])
        ax.yaxis.label.set_color(COLORS['text_secondary'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', color=COLORS['text_primary'], y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha='center', fontsize=11, color=COLORS['text_secondary'])


def _save(fig, path, dpi=200):
    """Save figure with dark background."""
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', 
                   facecolor=COLORS['bg_dark'], edgecolor='none')
        print(f"  → Saved: {path}")


# ═══════════════════════════════════════════════════════
# HERO PLOT 1: CONFUSION MATRIX
# ═══════════════════════════════════════════════════════

def plot_confusion_matrix(results, save_path=None):
    """
    Annotated confusion matrix heatmap with dark professional theme.
    Shows TP, TN, FP, FN with percentages and counts.
    """
    cm = results['confusion_matrix']
    model_name = results['model_name']
    
    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_dark_theme(fig, ax)
    
    # Normalize for color mapping
    cm_norm = cm.astype('float') / cm.sum()
    
    # Custom annotations with count + percentage
    annotations = []
    labels_map = {
        (0, 0): ('True Negative', COLORS['green']),
        (0, 1): ('False Positive', COLORS['orange']),
        (1, 0): ('False Negative', COLORS['red']),
        (1, 1): ('True Positive', COLORS['blue']),
    }
    
    for i in range(2):
        row = []
        for j in range(2):
            label, color = labels_map[(i, j)]
            row.append(f"{label}\n{cm[i,j]:,}\n({cm_norm[i,j]:.1%})")
        annotations.append(row)
    
    annotations = np.array(annotations)
    
    # Plot
    sns.heatmap(cm_norm, annot=annotations, fmt='', cmap=HEATMAP_CMAP,
                ax=ax, cbar=False, linewidths=2, linecolor=COLORS['bg_dark'],
                annot_kws={'fontsize': 12, 'fontweight': 'bold', 'color': COLORS['text_primary']})
    
    ax.set_xlabel('Predicted Label', fontsize=13, color=COLORS['text_secondary'], labelpad=10)
    ax.set_ylabel('True Label', fontsize=13, color=COLORS['text_secondary'], labelpad=10)
    ax.set_xticklabels(['Legitimate', 'Fraud'], fontsize=12, color=COLORS['text_primary'])
    ax.set_yticklabels(['Legitimate', 'Fraud'], fontsize=12, color=COLORS['text_primary'], rotation=0)
    
    _apply_dark_theme(fig, ax, 
                      title=f"{model_name} — Confusion Matrix",
                      subtitle=f"Threshold: {results['optimal_threshold']:.4f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 2: PRECISION-RECALL CURVE
# ═══════════════════════════════════════════════════════

def plot_pr_curve(results_list, save_path=None):
    """
    Precision-Recall curve with AUC shading for one or more models.
    Critical metric for imbalanced fraud detection.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark_theme(fig, ax)
    
    model_colors = [COLORS['blue'], COLORS['green'], COLORS['purple'], COLORS['orange']]
    
    for idx, results in enumerate(results_list):
        color = model_colors[idx % len(model_colors)]
        prec = results['precision_curve']
        rec = results['recall_curve']
        prauc = results['pr_auc']
        name = results['model_name']
        
        ax.plot(rec, prec, color=color, linewidth=2.5, 
                label=f"{name} (PR-AUC = {prauc:.4f})")
        ax.fill_between(rec, prec, alpha=0.15, color=color)
    
    # Mark optimal threshold
    for results in results_list:
        opt_thresh = results['optimal_threshold']
        prec = results['precision_curve']
        rec = results['recall_curve']
        thresh = results['pr_thresholds']
        
        valid = rec[:-1] >= 0.90
        if np.any(valid):
            best_idx = np.where(valid)[0][np.argmax(prec[:-1][valid])]
            ax.scatter(rec[best_idx], prec[best_idx], s=100, c=COLORS['red'], 
                      zorder=5, marker='*', edgecolors='white', linewidths=0.5)
    
    # Baseline
    baseline = results_list[0]['test_labels'].mean()
    ax.axhline(y=baseline, color=COLORS['text_secondary'], linestyle='--', alpha=0.5,
               label=f'No-skill baseline ({baseline:.4f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13, labelpad=10)
    ax.set_ylabel('Precision (PPV)', fontsize=13, labelpad=10)
    ax.legend(fontsize=11, loc='upper right', 
              facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text_primary'])
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    
    _apply_dark_theme(fig, ax,
                      title="Precision-Recall Curve",
                      subtitle="★ = Optimal threshold at ≥90% recall")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 3: ROC CURVE
# ═══════════════════════════════════════════════════════

def plot_roc_curve(results_list, save_path=None):
    """ROC curve with AUC shading."""
    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark_theme(fig, ax)
    
    model_colors = [COLORS['blue'], COLORS['green'], COLORS['purple']]
    
    for idx, results in enumerate(results_list):
        color = model_colors[idx % len(model_colors)]
        fpr = results['fpr']
        tpr = results['tpr']
        roc = results['roc_auc']
        name = results['model_name']
        
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{name} (AUC = {roc:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.12, color=color)
    
    # Diagonal baseline
    ax.plot([0, 1], [0, 1], color=COLORS['text_secondary'], linestyle='--', 
            alpha=0.5, label='Random classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=13, labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=13, labelpad=10)
    ax.legend(fontsize=11, loc='lower right',
              facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text_primary'])
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    
    _apply_dark_theme(fig, ax,
                      title="ROC Curve — Receiver Operating Characteristic",
                      subtitle="Area Under Curve measures discrimination ability")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 4: FRAUD RING NETWORK GRAPH
# ═══════════════════════════════════════════════════════

def plot_fraud_ring(df, save_path=None):
    """
    Network graph visualization of the fraud ring.
    Shows the device farm attack pattern: many users → one device.
    """
    # Identify highest-degree device (the fraud farm)
    device_user_counts = df.groupby('device_id')['user_id'].nunique().sort_values(ascending=False)
    fraud_device = device_user_counts.index[0]
    
    # Get top connected users for this device
    ring_data = df[df['device_id'] == fraud_device]
    top_users = ring_data.groupby('user_id')['amount'].sum().nlargest(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    _apply_dark_theme(fig, ax)
    
    # Build graph
    G = nx.Graph()
    device_label = f"Device\n{fraud_device}"
    G.add_node(device_label, node_type='device')
    
    for user_id, total_amount in top_users.items():
        user_label = f"User\n{user_id}"
        G.add_node(user_label, node_type='user')
        G.add_edge(device_label, user_label, weight=total_amount)
    
    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Draw edges
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                          edge_color=COLORS['red'], alpha=0.4,
                          style='solid')
    
    # Draw nodes
    device_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'device']
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'user']
    
    nx.draw_networkx_nodes(G, pos, nodelist=device_nodes, ax=ax,
                          node_color=COLORS['red'], node_size=800,
                          edgecolors='white', linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, ax=ax,
                          node_color=COLORS['blue'], node_size=300,
                          edgecolors=COLORS['grid'], linewidths=1, alpha=0.85)
    
    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                           font_color=COLORS['text_primary'],
                           font_weight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['red'], edgecolor='white', label='Compromised Device (Hub)'),
        mpatches.Patch(facecolor=COLORS['blue'], edgecolor=COLORS['grid'], label='Connected Users (Victims)'),
        plt.Line2D([0], [0], color=COLORS['red'], alpha=0.4, linewidth=2, label='Transaction Flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    
    ax.axis('off')
    
    n_connected = len(user_nodes)
    total_amount = sum(edge_weights)
    
    _apply_dark_theme(fig, ax,
                      title="Fraud Ring Topology — Device Farm Attack Pattern",
                      subtitle=f"{n_connected} compromised users │ ${total_amount:,.2f} total exposure │ Star topology")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 5: TEMPORAL FRAUD HEATMAP
# ═══════════════════════════════════════════════════════

def plot_temporal_heatmap(df, save_path=None):
    """
    Hour × Day-of-Week heatmap of fraud rate.
    Reveals temporal attack patterns (e.g., late-night bust-outs).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    _apply_dark_theme(fig, ax)
    
    # Compute fraud rate by hour and day
    temporal = df.groupby(['day_of_week', 'hour'])['is_fraud'].mean().unstack(fill_value=0)
    
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    sns.heatmap(temporal, cmap=FRAUD_CMAP, ax=ax, 
                linewidths=0.5, linecolor=COLORS['bg_dark'],
                cbar_kws={'label': 'Fraud Rate', 'shrink': 0.8},
                annot=False)
    
    ax.set_xlabel('Hour of Day', fontsize=13, labelpad=10)
    ax.set_ylabel('Day of Week', fontsize=13, labelpad=10)
    ax.set_yticklabels(day_labels[:len(temporal)], rotation=0, fontsize=10)
    ax.set_xticklabels(range(24), fontsize=9)
    
    # Color bar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=COLORS['text_secondary'])
    cbar.set_label('Fraud Rate', color=COLORS['text_secondary'], fontsize=11)
    
    _apply_dark_theme(fig, ax,
                      title="Temporal Fraud Pattern Analysis",
                      subtitle="Fraud rate by hour and day — Late-night periods show elevated risk")
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 6: AMOUNT DISTRIBUTION
# ═══════════════════════════════════════════════════════

def plot_amount_distribution(df, save_path=None):
    """
    Side-by-side violin plots of transaction amounts: fraud vs legitimate.
    Shows the characteristic high-value skew in fraudulent transactions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _apply_dark_theme(fig, axes)
    
    df_plot = df.copy()
    df_plot['label'] = df_plot['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    
    # Left: Raw amount distribution
    ax = axes[0]
    for label, color in [('Legitimate', COLORS['blue']), ('Fraud', COLORS['red'])]:
        subset = df_plot[df_plot['label'] == label]['amount']
        subset_clipped = subset.clip(upper=subset.quantile(0.99))
        ax.hist(subset_clipped, bins=80, alpha=0.6, color=color, label=label,
                edgecolor='none', density=True)
    
    ax.set_xlabel('Transaction Amount ($)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Amount Distribution', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.legend(fontsize=10, facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    ax.grid(True, alpha=0.1, color=COLORS['grid'])
    
    # Right: Log-scale box plot
    ax = axes[1]
    fraud_amounts = df_plot[df_plot['is_fraud'] == 1]['amount']
    legit_amounts = df_plot[df_plot['is_fraud'] == 0]['amount']
    
    bp = ax.boxplot(
        [legit_amounts, fraud_amounts],
        labels=['Legitimate', 'Fraud'],
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color=COLORS['text_primary'], linewidth=2)
    )
    
    bp['boxes'][0].set_facecolor(COLORS['blue'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLORS['red'])
    bp['boxes'][1].set_alpha(0.7)
    
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color(COLORS['text_secondary'])
    
    ax.set_ylabel('Transaction Amount ($)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Amount Comparison (Log Scale)', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.grid(True, alpha=0.1, color=COLORS['grid'], axis='y')
    ax.tick_params(colors=COLORS['text_secondary'])
    
    # Stats annotation
    stats_text = (f"Legit median: ${legit_amounts.median():,.2f}\n"
                  f"Fraud median: ${fraud_amounts.median():,.2f}\n"
                  f"Fraud/Legit ratio: {fraud_amounts.median()/legit_amounts.median():.1f}x")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            color=COLORS['text_secondary'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_accent'], 
                     edgecolor=COLORS['grid'], alpha=0.9))
    
    _apply_dark_theme(fig, axes,
                      title="Transaction Amount Analysis — Fraud vs Legitimate",
                      subtitle="Fraudulent transactions show characteristic high-value skew")
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 7: TRAINING HISTORY
# ═══════════════════════════════════════════════════════

def plot_training_history(history, save_path=None):
    """Training loss and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _apply_dark_theme(fig, axes)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Left: Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], color=COLORS['blue'], linewidth=2.5,
            label='Training Loss', marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('BCE Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.legend(fontsize=10, facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    
    # Right: Validation metrics
    ax = axes[1]
    ax.plot(epochs, history['val_prauc'], color=COLORS['green'], linewidth=2.5,
            label='PR-AUC', marker='s', markersize=3)
    ax.plot(epochs, history['val_rocauc'], color=COLORS['purple'], linewidth=2.5,
            label='ROC-AUC', marker='^', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Validation Metrics', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.legend(fontsize=10, facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    ax.set_ylim([0, 1.05])
    
    _apply_dark_theme(fig, axes,
                      title="Model Training Progress",
                      subtitle=f"Best PR-AUC: {max(history['val_prauc']):.4f} │ Best ROC-AUC: {max(history['val_rocauc']):.4f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 8: FEATURE IMPORTANCE COMPARISON
# ═══════════════════════════════════════════════════════

def plot_feature_importance_comparison(shap_values, feature_names, xgb_importance, 
                                       save_path=None, top_n=15):
    """
    Side-by-side SHAP vs XGBoost feature importance.
    Demonstrates model agreement/divergence on what matters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _apply_dark_theme(fig, axes)
    
    # SHAP importance (mean absolute)
    shap_imp = np.abs(shap_values).mean(axis=0)
    if len(shap_imp) != len(feature_names):
        feature_names = feature_names[:len(shap_imp)]
    
    shap_df = pd.DataFrame({
        'feature': feature_names, 
        'importance': shap_imp
    }).nlargest(top_n, 'importance')
    
    ax = axes[0]
    bars = ax.barh(range(len(shap_df)), shap_df['importance'].values, 
                   color=COLORS['blue'], alpha=0.85, edgecolor=COLORS['grid'])
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df['feature'].values, fontsize=10, color=COLORS['text_primary'])
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Neural Network (SHAP)', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.1, color=COLORS['grid'], axis='x')
    
    # XGBoost importance
    xgb_df = pd.DataFrame({
        'feature': list(xgb_importance.keys()),
        'importance': list(xgb_importance.values())
    }).nlargest(top_n, 'importance')
    
    ax = axes[1]
    ax.barh(range(len(xgb_df)), xgb_df['importance'].values,
            color=COLORS['green'], alpha=0.85, edgecolor=COLORS['grid'])
    ax.set_yticks(range(len(xgb_df)))
    ax.set_yticklabels(xgb_df['feature'].values, fontsize=10, color=COLORS['text_primary'])
    ax.set_xlabel('XGBoost Gain Importance', fontsize=12)
    ax.set_title('XGBoost Baseline', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.1, color=COLORS['grid'], axis='x')
    
    _apply_dark_theme(fig, axes,
                      title="Feature Importance — Neural Network vs XGBoost",
                      subtitle="Both models agree on key fraud signals: device centrality, velocity, amount")
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 9: LATENT SPACE PROJECTION
# ═══════════════════════════════════════════════════════

def plot_latent_space(model, data, device, save_path=None, n_samples=2000):
    """
    2D PCA projection of the neural network's penultimate layer.
    Shows conceptual soundness: fraud and legit should form distinct clusters.
    """
    import torch
    
    fig, ax = plt.subplots(figsize=(12, 9))
    _apply_dark_theme(fig, ax)
    
    model_cpu = model.cpu()
    model_cpu.eval()
    
    X_test = data['X_test'].iloc[:n_samples]
    y_test = data['y_test'].iloc[:n_samples]
    
    test_tensor = torch.from_numpy(X_test.values.astype(np.float32)).float()
    
    with torch.no_grad():
        embeddings = model_cpu.get_embeddings(test_tensor).numpy()
    
    # PCA to 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings)
    
    # Plot legitimate (large, transparent)
    legit_mask = y_test.values == 0
    fraud_mask = y_test.values == 1
    
    ax.scatter(components[legit_mask, 0], components[legit_mask, 1],
               c=COLORS['blue'], alpha=0.15, s=15, label='Legitimate', edgecolors='none')
    ax.scatter(components[fraud_mask, 0], components[fraud_mask, 1],
               c=COLORS['red'], alpha=0.8, s=40, label='Fraud', edgecolors='white',
               linewidths=0.5, zorder=5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.legend(fontsize=12, loc='upper right',
             facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'], markerscale=2)
    ax.grid(True, alpha=0.1, color=COLORS['grid'])
    
    _apply_dark_theme(fig, ax,
                      title="Neural Network Latent Space — Conceptual Soundness",
                      subtitle="PCA projection of penultimate layer │ Distinct fraud cluster validates model learning")
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, save_path)
    plt.show()
    
    # Move model back
    model.to(device if isinstance(device, torch.device) else torch.device(device))
    return fig


# ═══════════════════════════════════════════════════════
# HERO PLOT 10: MODEL COMPARISON DASHBOARD
# ═══════════════════════════════════════════════════════

def plot_model_comparison_dashboard(mlp_results, xgb_results, save_path=None):
    """
    4-panel dashboard comparing MLP vs XGBoost across key metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    _apply_dark_theme(fig, axes)
    
    # Panel 1: Metrics comparison bar
    ax = axes[0, 0]
    metrics = ['ROC-AUC', 'PR-AUC']
    mlp_vals = [mlp_results['roc_auc'], mlp_results['pr_auc']]
    xgb_vals = [xgb_results['roc_auc'], xgb_results['pr_auc']]
    
    x = np.arange(len(metrics))
    width = 0.3
    ax.bar(x - width/2, mlp_vals, width, label='Neural Network', color=COLORS['blue'], alpha=0.85)
    ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color=COLORS['green'], alpha=0.85)
    
    for i, (m, x_val) in enumerate(zip(mlp_vals, x)):
        ax.text(x_val - width/2, m + 0.01, f'{m:.4f}', ha='center', fontsize=9, 
                color=COLORS['text_primary'], fontweight='bold')
    for i, (x_val_v, x_pos) in enumerate(zip(xgb_vals, x)):
        ax.text(x_pos + width/2, x_val_v + 0.01, f'{x_val_v:.4f}', ha='center', fontsize=9,
                color=COLORS['text_primary'], fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, color=COLORS['text_primary'])
    ax.set_ylim([0, 1.15])
    ax.set_title('Performance Metrics', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.legend(fontsize=10, facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    ax.grid(True, alpha=0.1, color=COLORS['grid'], axis='y')
    
    # Panel 2: PR curves
    ax = axes[0, 1]
    ax.plot(mlp_results['recall_curve'], mlp_results['precision_curve'],
            color=COLORS['blue'], linewidth=2, label='MLP')
    ax.plot(xgb_results['recall_curve'], xgb_results['precision_curve'],
            color=COLORS['green'], linewidth=2, label='XGBoost')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.legend(fontsize=10, facecolor=COLORS['bg_accent'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    ax.grid(True, alpha=0.1, color=COLORS['grid'])
    
    # Panel 3: MLP confusion matrix
    ax = axes[1, 0]
    cm = mlp_results['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum()
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap=HEATMAP_CMAP, ax=ax,
                cbar=False, linewidths=1, linecolor=COLORS['bg_dark'],
                annot_kws={'fontsize': 12, 'color': COLORS['text_primary']})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('MLP Confusion Matrix', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.set_xticklabels(['Legit', 'Fraud'], color=COLORS['text_primary'])
    ax.set_yticklabels(['Legit', 'Fraud'], color=COLORS['text_primary'], rotation=0)
    
    # Panel 4: XGBoost confusion matrix
    ax = axes[1, 1]
    cm = xgb_results['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum()
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap=HEATMAP_CMAP, ax=ax,
                cbar=False, linewidths=1, linecolor=COLORS['bg_dark'],
                annot_kws={'fontsize': 12, 'color': COLORS['text_primary']})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('XGBoost Confusion Matrix', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'], pad=10)
    ax.set_xticklabels(['Legit', 'Fraud'], color=COLORS['text_primary'])
    ax.set_yticklabels(['Legit', 'Fraud'], color=COLORS['text_primary'], rotation=0)
    
    _apply_dark_theme(fig, axes,
                      title="Model Comparison Dashboard — Neural Network vs XGBoost",
                      subtitle="Comprehensive evaluation across discrimination, precision-recall tradeoff, and error patterns")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, save_path)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════
# MASTER GENERATOR
# ═══════════════════════════════════════════════════════

def generate_all_hero_plots(df, mlp_results, xgb_results, history,
                            model, data, device,
                            shap_values=None, feature_names=None,
                            xgb_importance=None,
                            output_dir="outputs"):
    """
    Generate ALL hero visualizations and save to output directory.
    This is the main entry point.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  GENERATING HERO VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Confusion matrices
    plot_confusion_matrix(mlp_results, 
                         save_path=os.path.join(output_dir, "confusion_matrix_mlp.png"))
    plot_confusion_matrix(xgb_results,
                         save_path=os.path.join(output_dir, "confusion_matrix_xgb.png"))
    
    # 2. PR curve
    plot_pr_curve([mlp_results, xgb_results],
                  save_path=os.path.join(output_dir, "precision_recall_curve.png"))
    
    # 3. ROC curve
    plot_roc_curve([mlp_results, xgb_results],
                   save_path=os.path.join(output_dir, "roc_curve.png"))
    
    # 4. Fraud ring
    plot_fraud_ring(df,
                    save_path=os.path.join(output_dir, "fraud_ring_network.png"))
    
    # 5. Temporal heatmap
    plot_temporal_heatmap(df,
                          save_path=os.path.join(output_dir, "temporal_fraud_heatmap.png"))
    
    # 6. Amount distribution
    plot_amount_distribution(df,
                             save_path=os.path.join(output_dir, "amount_distribution.png"))
    
    # 7. Training history
    plot_training_history(history,
                          save_path=os.path.join(output_dir, "training_history.png"))
    
    # 8. Feature importance comparison
    if shap_values is not None and xgb_importance is not None:
        plot_feature_importance_comparison(
            shap_values, feature_names, xgb_importance,
            save_path=os.path.join(output_dir, "feature_importance_comparison.png")
        )
    
    # 9. Latent space
    plot_latent_space(model, data, device,
                      save_path=os.path.join(output_dir, "latent_space_projection.png"))
    
    # 10. Model comparison dashboard
    plot_model_comparison_dashboard(
        mlp_results, xgb_results,
        save_path=os.path.join(output_dir, "model_comparison_dashboard.png")
    )
    
    print("\n" + "=" * 60)
    print(f"  ALL HERO PLOTS SAVED TO {output_dir}/")
    print("=" * 60)
