#!/usr/bin/env python3
"""
Train Failure Prediction Classifier

Tiny MLP to predict SAFE (0) vs UNSAFE (1) from a window of k=8 timesteps.
Input: 96 features (8 timesteps × 12 features)
Output: Binary classification

Architecture: 96 → 64 → 32 → 2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/data/windowed_dataset.npz")
OUTPUT_DIR = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models")
PLOTS_DIR = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/plots")

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 15  # Early stopping patience
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class FailureClassifier(nn.Module):
    """Tiny MLP for failure prediction."""
    
    def __init__(self, input_dim=96, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # 2 classes: SAFE, UNSAFE
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def load_data():
    """Load windowed dataset."""
    data = np.load(DATA_PATH)
    
    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.LongTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.LongTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"])
    y_test = torch.LongTensor(data["y_test"])
    class_weight = float(data["class_weight"][0])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_weight


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(UNSAFE)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    
    # AUROC (handle edge case where only one class present)
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_probs)
    else:
        auroc = 0.5
    
    return total_loss / len(all_labels), acc, auroc, all_probs, all_labels


def find_threshold_at_fpr(probs, labels, target_fpr=0.10):
    """Find threshold that achieves target FPR."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Find threshold closest to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    return thresholds[idx], tpr[idx], fpr[idx]


def main():
    print("=" * 60)
    print("TRAINING FAILURE PREDICTION CLASSIFIER")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    
    # Load data
    print("\n📂 Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, class_weight = load_data()
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Class weight: {class_weight:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    
    # Create model
    print("\n🧠 Creating model...")
    model = FailureClassifier(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.3)
    model = model.to(DEVICE)
    print(f"   Architecture: {X_train.shape[1]} → 64 → 32 → 2")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights
    weights = torch.FloatTensor([1.0, class_weight]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n🚀 Training...")
    best_val_auroc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_auroc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        
        if epoch % 10 == 0 or val_auroc > best_val_auroc:
            print(f"   Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_AUROC={val_auroc:.4f}")
        
        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            # Save best model
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n   Early stopping at epoch {epoch}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_loss, test_acc, test_auroc, test_probs, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    
    # Find threshold at 10% FPR
    threshold, recall_at_fpr, actual_fpr = find_threshold_at_fpr(test_probs, test_labels, target_fpr=0.10)
    print(f"\n   At FPR ≈ {actual_fpr:.1%}:")
    print(f"     Threshold: {threshold:.3f}")
    print(f"     Recall: {recall_at_fpr:.1%}")
    
    # Confusion matrix at optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    test_preds = (test_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(test_labels, test_preds)
    
    print(f"\n   Confusion Matrix (threshold={optimal_threshold:.3f}):")
    print(f"                    Predicted")
    print(f"                  SAFE  UNSAFE")
    print(f"     Actual SAFE  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"     Actual UNSAFE{cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Save results
    results = {
        "test_auroc": float(test_auroc),
        "test_accuracy": float(test_acc),
        "optimal_threshold": float(optimal_threshold),
        "recall_at_10pct_fpr": float(recall_at_fpr),
        "epochs_trained": epoch + 1,
        "best_val_auroc": float(best_val_auroc)
    }
    
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {test_auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.scatter([actual_fpr], [recall_at_fpr], color='red', s=100, zorder=5, label=f'@10% FPR: {recall_at_fpr:.1%} recall')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve - Failure Prediction')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train')
    plt.plot(history["val_loss"], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_auroc"])
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Validation AUROC')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Model saved to: {OUTPUT_DIR / 'best_model.pt'}")
    print(f"✅ Results saved to: {OUTPUT_DIR / 'results.json'}")
    print(f"✅ Plots saved to: {PLOTS_DIR}")
    
    return model, results


if __name__ == "__main__":
    main()
