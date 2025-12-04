# Quick Start Guide

This guide will help you get started with ACIE quickly.

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```python
import torch
from acie import ACIE_Core

print("ACIE successfully installed!")
```

## Your First Model

### 1. Train a Simple Model

```bash
python scripts/train.py --epochs 10 --experiment-name quickstart
```

This will:
- Create synthetic training data
- Train ACIE for 10 epochs
- Save the model to `models/quickstart/`

### 2. Evaluate the Model

```bash
python scripts/evaluate.py \
    --model-path models/quickstart/best_model.pth \
    --experiment-name quickstart_eval
```

This will:
- Load your trained model
- Evaluate on test data
- Generate visualizations in `results/quickstart_eval/`

### 3. Run Inference

```bash
python scripts/inference.py \
    --model-path models/quickstart/best_model.pth \
    --interactive
```

Try these commands:
- Type `random` to test with random data
- Type `quit` to exit

## Understanding the Output

### Training Output

```
Epoch 1/10 - Train Loss: 1.2345 - DAG: 0.0234
Val Loss: 1.1234 - Val Acc: 0.7500
```

- **Train Loss**: Combined loss (policy + DAG constraint + sparsity)
- **DAG**: Acyclicity constraint violation (should decrease)
- **Val Acc**: Validation accuracy

### Evaluation Output

Check `results/quickstart_eval/` for:

1. **causal_graph.png** - Learned causal relationships
2. **confusion_matrix.png** - Classification performance
3. **entropy_distribution.png** - Information filter analysis
4. **metrics.json** - Detailed performance metrics

## Next Steps

1. **Customize your model**: Edit `configs/default_config.json`
2. **Use your own data**: Implement `RealWorldDataset` in `acie/dataset.py`
3. **Tune hyperparameters**: Adjust learning rate, lambda values, etc.
4. **Run adversarial tests**: Add `--adversarial-test` to evaluation

## Common Issues

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/train.py --batch-size 16
```

### Slow Training

Use GPU if available:
```bash
python scripts/train.py --device cuda
```

### DAG Constraint Not Converging

Increase lambda_dag:
```bash
python scripts/train.py --lambda-dag 0.5
```

## Need Help?

- Check the full README.md for detailed documentation
- Review example notebooks in `notebooks/`
- Open an issue on GitHub
