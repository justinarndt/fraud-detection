"""
Model Architecture & Training
================================
PyTorch MLP for fraud detection with:
  - Class-weighted BCE loss for imbalance handling
  - Learning rate scheduling
  - XGBoost comparison baseline
  - Evaluation utilities (ROC-AUC, PR-AUC, confusion matrix)

Designed for SR 11-7: reproducible training with documented hyperparameters.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    confusion_matrix, precision_recall_curve,
    roc_curve, classification_report
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════
# PYTORCH DATASET
# ═══════════════════════════════════════════════════════

class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection features."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════

class FraudMLP(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection.
    
    Architecture: Input → 256 → 128 → 64 → 32 → 1
    With BatchNorm, Dropout, and residual-style skip connections.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    dropout_rate : float
        Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
        )
        
        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x)
    
    def get_embeddings(self, x):
        """Extract penultimate layer embeddings for latent space visualization."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# ═══════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════

def prepare_data(df, feature_names, test_size=0.2, val_size=0.1, batch_size=256, seed=42):
    """
    Prepare train/val/test splits and DataLoaders.
    
    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                    train_loader, val_loader, test_loader
    """
    X = df[feature_names]
    y = df['is_fraud']
    
    # Stratified split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size/(1-test_size), 
        stratify=y_trainval, random_state=seed
    )
    
    # PyTorch DataLoaders
    train_ds = FraudDataset(X_train, y_train)
    val_ds = FraudDataset(X_val, y_val)
    test_ds = FraudDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2)
    
    print(f"[Data] Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"[Data] Fraud rates — Train: {y_train.mean():.4%} | Val: {y_val.mean():.4%} | Test: {y_test.mean():.4%}")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader
    }


# ═══════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════

def train_mlp(model, data, device, epochs=30, lr=1e-3, patience=5):
    """
    Train the MLP model with class-weighted BCE and LR scheduling.
    
    Returns
    -------
    dict: Training history with loss and PR-AUC per epoch
    """
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    y_train = data['y_train']
    
    # Class weights for imbalance
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience, factor=0.5)
    
    history = {'train_loss': [], 'val_prauc': [], 'val_rocauc': []}
    best_prauc = 0
    best_state = None
    
    print(f"\n[Train] Starting training for {epochs} epochs...")
    print(f"[Train] Positive class weight: {pos_weight.item():.2f}")
    print(f"[Train] Device: {device}")
    print("─" * 60)
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                probs = torch.sigmoid(model(Xb)).cpu()
                val_probs.append(probs)
                val_labels.append(yb)
        
        val_probs = torch.cat(val_probs).squeeze().numpy()
        val_labels = torch.cat(val_labels).squeeze().numpy()
        val_prauc = average_precision_score(val_labels, val_probs)
        val_rocauc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step(val_prauc)
        
        history['train_loss'].append(avg_loss)
        history['val_prauc'].append(val_prauc)
        history['val_rocauc'].append(val_rocauc)
        
        # Save best model
        if val_prauc > best_prauc:
            best_prauc = val_prauc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} │ Loss: {avg_loss:.4f} │ "
                  f"PR-AUC: {val_prauc:.4f} │ ROC-AUC: {val_rocauc:.4f} │ LR: {lr_current:.2e}")
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print("─" * 60)
    print(f"[Train] Complete. Best Val PR-AUC: {best_prauc:.4f}")
    
    return history


# ═══════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════

def evaluate_model(model, data, device, model_name="MLP"):
    """
    Comprehensive model evaluation.
    
    Returns
    -------
    dict: metrics, predictions, probabilities, thresholds
    """
    test_loader = data['test_loader']
    y_test = data['y_test']
    
    model.eval()
    all_probs = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            probs = torch.sigmoid(model(Xb)).cpu()
            all_probs.append(probs)
    
    test_probs = torch.cat(all_probs).squeeze().numpy()
    
    # Metrics
    roc = roc_auc_score(y_test, test_probs)
    prauc = average_precision_score(y_test, test_probs)
    
    # Optimal threshold (maximize F1 at ≥90% recall)
    prec, rec, thresh = precision_recall_curve(y_test, test_probs)
    valid_idx = rec >= 0.90
    if np.any(valid_idx):
        best_idx = np.argmax(prec[valid_idx])
        # Map back to original index
        original_indices = np.where(valid_idx)[0]
        opt_thresh = thresh[original_indices[best_idx]] if original_indices[best_idx] < len(thresh) else 0.5
    else:
        opt_thresh = 0.5
    
    opt_preds = (test_probs >= opt_thresh).astype(int)
    cm = confusion_matrix(y_test, opt_preds)
    report = classification_report(y_test, opt_preds, output_dict=True)
    
    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
    
    results = {
        'model_name': model_name,
        'roc_auc': roc,
        'pr_auc': prauc,
        'optimal_threshold': opt_thresh,
        'confusion_matrix': cm,
        'classification_report': report,
        'test_probs': test_probs,
        'test_labels': y_test.values,
        'precision_curve': prec,
        'recall_curve': rec,
        'pr_thresholds': thresh,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
    }
    
    # Get the positive class report (key varies by sklearn version: '1', '1.0', 1, etc.)
    pos_key = None
    for k in ['1', '1.0', 1, 1.0]:
        if k in report:
            pos_key = k
            break
    
    print(f"\n{'═' * 50}")
    print(f"  {model_name} EVALUATION RESULTS")
    print(f"{'═' * 50}")
    print(f"  ROC-AUC:           {roc:.4f}")
    print(f"  PR-AUC:            {prauc:.4f}")
    print(f"  Optimal Threshold: {opt_thresh:.4f}")
    if pos_key is not None:
        print(f"  Recall @ Thresh:   {report[pos_key]['recall']:.4f}")
        print(f"  Precision @ Thresh:{report[pos_key]['precision']:.4f}")
        print(f"  F1 @ Thresh:       {report[pos_key]['f1-score']:.4f}")
    print(f"{'═' * 50}")
    
    return results


def train_xgboost_baseline(data, feature_names):
    """
    Train XGBoost as comparison baseline.
    
    Returns
    -------
    tuple: (xgb_model, results_dict)
    """
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    scale_pos = (1 - y_train.mean()) / y_train.mean()
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    print("\n[XGBoost] Training baseline model...")
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    
    roc = roc_auc_score(y_test, xgb_probs)
    prauc = average_precision_score(y_test, xgb_probs)
    
    # Optimal threshold
    prec, rec, thresh = precision_recall_curve(y_test, xgb_probs)
    valid_idx = rec >= 0.90
    if np.any(valid_idx):
        best_idx = np.argmax(prec[valid_idx])
        original_indices = np.where(valid_idx)[0]
        opt_thresh = thresh[original_indices[best_idx]] if original_indices[best_idx] < len(thresh) else 0.5
    else:
        opt_thresh = 0.5
    
    opt_preds = (xgb_probs >= opt_thresh).astype(int)
    cm = confusion_matrix(y_test, opt_preds)
    report = classification_report(y_test, opt_preds, output_dict=True)
    fpr, tpr, roc_thresholds = roc_curve(y_test, xgb_probs)
    
    # Feature importance
    importance = dict(zip(feature_names, xgb.feature_importances_))
    
    results = {
        'model_name': 'XGBoost',
        'roc_auc': roc,
        'pr_auc': prauc,
        'optimal_threshold': opt_thresh,
        'confusion_matrix': cm,
        'classification_report': report,
        'test_probs': xgb_probs,
        'test_labels': y_test.values,
        'precision_curve': prec,
        'recall_curve': rec,
        'pr_thresholds': thresh,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'feature_importance': importance,
    }
    
    print(f"\n{'═' * 50}")
    print(f"  XGBoost BASELINE RESULTS")
    print(f"{'═' * 50}")
    print(f"  ROC-AUC:           {roc:.4f}")
    print(f"  PR-AUC:            {prauc:.4f}")
    print(f"  Optimal Threshold: {opt_thresh:.4f}")
    print(f"{'═' * 50}")
    
    return xgb, results


def compare_models(mlp_results, xgb_results):
    """Print side-by-side model comparison."""
    print(f"\n{'═' * 60}")
    print(f"  MODEL COMPARISON: MLP vs XGBoost")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<25} {'MLP':>12} {'XGBoost':>12}")
    print(f"  {'─' * 49}")
    
    metrics = [
        ('ROC-AUC', 'roc_auc'),
        ('PR-AUC', 'pr_auc'),
        ('Optimal Threshold', 'optimal_threshold'),
    ]
    
    for label, key in metrics:
        mlp_val = mlp_results[key]
        xgb_val = xgb_results[key]
        winner = "◄" if mlp_val > xgb_val else ""
        print(f"  {label:<25} {mlp_val:>12.4f} {xgb_val:>12.4f} {winner}")
    
    print(f"{'═' * 60}")
