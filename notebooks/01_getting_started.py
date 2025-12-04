# Example Jupyter Notebook - Getting Started with ACIE
# This notebook demonstrates basic usage of the ACIE framework

## Installation Check
import sys
print(f"Python version: {sys.version}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Import ACIE
from acie import ACIE_Core, ACIETrainer, ACIEEvaluator
print("[+] ACIE imported successfully!")

## 1. Create a Simple Model
model = ACIE_Core(
    input_dim=100,
    causal_nodes=10,
    action_space=5
)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

## 2. Generate Sample Data
from acie.dataset import CyberEventDataset
from torch.utils.data import DataLoader

dataset = CyberEventDataset(
    num_samples=200,
    input_dim=100,
    num_classes=5
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"Dataset created with {len(dataset)} samples")

## 3. Test Forward Pass
sample_batch, labels = next(iter(train_loader))
output = model(sample_batch)

if output == "NO_THREAT":
    print("Output: NO_THREAT (low entropy)")
else:
    action_probs, adjacency = output
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Adjacency matrix shape: {adjacency.shape}")

## 4. Visualize Causal Graph
import matplotlib.pyplot as plt
import seaborn as sns

adjacency_matrix = model.adjacency.detach().numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(adjacency_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Initial Causal Graph (Before Training)")
plt.xlabel("Effect Node")
plt.ylabel("Cause Node")
plt.show()

## 5. Quick Training Demo
from acie.utils import set_seed

set_seed(42)

trainer = ACIETrainer(model, learning_rate=1e-3)

# Train for just 5 epochs as demo
history = trainer.fit(train_loader, epochs=5)

print("Training complete!")

## 6. Plot Training Loss
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.show()

## 7. Check Learned Causal Graph
adjacency_trained = model.adjacency.detach().numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(adjacency_trained, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Learned Causal Graph (After Training)")
plt.xlabel("Effect Node")
plt.ylabel("Cause Node")
plt.show()

## Summary
print("=" * 50)
print("ACIE Quick Demo Complete!")
print("=" * 50)
print(f"Initial Loss: {history['train_loss'][0]:.4f}")
print(f"Final Loss: {history['train_loss'][-1]:.4f}")
print(f"Improvement: {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}%")
