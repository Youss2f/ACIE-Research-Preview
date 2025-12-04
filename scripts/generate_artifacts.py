"""
Generate Research Artifacts for Public Release

This script programmatically generates the "money shot" visualizations:
1. Causal Graph (learned adjacency matrix as directed graph)
2. Robustness Plot (clean vs adversarial accuracy comparison)

Output artifacts are saved to docs/assets/ for inclusion in RESEARCH_PREVIEW.md
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import DataLoader, random_split

# ACIE imports
from acie import ACIE_Core, ACIETrainer, ACIERecorder, ACIEConfig
from acie.dataset import CyberLogDataset
from acie.utils import set_seed, get_device


def generate_causal_graph(model: ACIE_Core, output_path: Path, threshold: float = 0.1):
    """
    Generate causal graph visualization from learned adjacency matrix.
    
    Args:
        model: Trained ACIE_Core model
        output_path: Path to save the PNG file
        threshold: Edge weight threshold for visualization
    """
    # Extract adjacency matrix
    adjacency = model.adjacency.detach().cpu().numpy()
    adjacency_thresholded = np.where(np.abs(adjacency) > threshold, adjacency, 0)
    
    # Create directed graph
    G = nx.DiGraph()
    for i in range(adjacency_thresholded.shape[0]):
        G.add_node(i, label=f"N{i}")
        for j in range(adjacency_thresholded.shape[1]):
            if adjacency_thresholded[i, j] != 0:
                G.add_edge(i, j, weight=adjacency_thresholded[i, j])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACIE: Learned Causal Structure", fontsize=14, fontweight='bold')
    
    # Heatmap
    im = axes[0].imshow(adjacency, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[0].set_title("Adjacency Matrix $A$", fontsize=12)
    axes[0].set_xlabel("Target Node $j$")
    axes[0].set_ylabel("Source Node $i$")
    axes[0].set_xticks(range(adjacency.shape[0]))
    axes[0].set_yticks(range(adjacency.shape[1]))
    plt.colorbar(im, ax=axes[0], label="Edge Weight $A_{ij}$", shrink=0.8)
    
    # Graph
    pos = nx.spring_layout(G, seed=42, k=2)
    
    if G.number_of_edges() > 0:
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        edge_colors = ["#d62728" if w < 0 else "#1f77b4" for w in edge_weights]
        edge_widths = [abs(w) * 3 + 0.5 for w in edge_weights]
    else:
        edge_colors = []
        edge_widths = []
    
    nx.draw(
        G, pos, ax=axes[1],
        node_color="#7fcdbb",
        node_size=600,
        with_labels=True,
        font_size=10,
        font_weight='bold',
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle="arc3,rad=0.1"
    )
    axes[1].set_title(f"Causal Graph (threshold={threshold})", fontsize=12)
    
    # Add legend
    axes[1].plot([], [], color="#1f77b4", linewidth=2, label="Positive causation")
    axes[1].plot([], [], color="#d62728", linewidth=2, label="Negative causation")
    axes[1].legend(loc='upper right', fontsize=9)
    
    # Stats annotation
    stats_text = f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}\nDAG: {nx.is_directed_acyclic_graph(G)}"
    axes[1].annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                     fontsize=9, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved causal graph to: {output_path}")


def fgsm_attack(model, data, target, epsilon=0.1):
    """Fast Gradient Sign Method attack."""
    data = data.clone().detach().requires_grad_(True)
    
    output = model(data)
    if output == "NO_THREAT":
        return data.detach()
    
    action_probs, _ = output
    loss = torch.nn.CrossEntropyLoss()(action_probs, target)
    
    model.zero_grad()
    loss.backward()
    
    perturbation = epsilon * data.grad.sign()
    perturbed_data = data + perturbation
    
    return perturbed_data.detach()


def evaluate_robustness(model, test_loader, device, epsilons=[0.0, 0.05, 0.1, 0.15, 0.2]):
    """Evaluate model accuracy across different perturbation strengths."""
    accuracies = []
    
    for epsilon in epsilons:
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if epsilon > 0:
                model.train()
                data = fgsm_attack(model, data, target, epsilon)
                model.eval()
            
            with torch.no_grad():
                output = model(data)
                if output != "NO_THREAT":
                    probs, _ = output
                    pred = probs.argmax(dim=1)
                    correct += (pred == target).sum().item()
            
            total += target.size(0)
        
        acc = correct / total if total > 0 else 0
        accuracies.append(acc)
        print(f"  ε={epsilon:.2f}: Accuracy={acc:.4f}")
    
    return epsilons, accuracies


def generate_robustness_plot(epsilons, accuracies, output_path: Path):
    """Generate robustness plot showing accuracy vs perturbation strength."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot accuracy curve
    ax.plot(epsilons, accuracies, 'o-', color='#2c7bb6', linewidth=2, markersize=8, label='ACIE Model')
    ax.fill_between(epsilons, accuracies, alpha=0.3, color='#2c7bb6')
    
    # Reference line (random baseline)
    ax.axhline(y=0.2, color='#d7191c', linestyle='--', linewidth=1.5, label='Random Baseline (1/5)')
    
    # Formatting
    ax.set_xlabel('Perturbation Strength (ε)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('ACIE: Adversarial Robustness Evaluation', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.01, max(epsilons) + 0.01)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add annotation for robustness gap
    clean_acc = accuracies[0]
    worst_acc = min(accuracies)
    gap = clean_acc - worst_acc
    
    annotation_text = f"Clean: {clean_acc:.1%}\nWorst: {worst_acc:.1%}\nGap: {gap:.1%}"
    ax.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved robustness plot to: {output_path}")


def main():
    print("=" * 60)
    print("ACIE Artifact Generation for Public Release")
    print("=" * 60)
    
    # Setup
    set_seed(42)
    device = get_device(prefer_gpu=True)
    output_dir = project_root / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = ACIEConfig(experiment_name="artifact_generation")
    config.model.input_dim = 100
    config.model.causal_nodes = 10
    config.model.action_space = 5
    config.training.epochs = 15
    config.training.batch_size = 32
    config.data.num_samples = 500
    
    print(f"\nDevice: {device}")
    print(f"Output directory: {output_dir}")
    
    # Create dataset
    print("\n[1/4] Creating CyberLogDataset...")
    dataset = CyberLogDataset(
        num_samples=config.data.num_samples,
        input_dim=config.model.input_dim,
        num_classes=config.model.action_space,
        seed=config.seed
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create and train model
    print("\n[2/4] Training ACIE model...")
    model = ACIE_Core(
        input_dim=config.model.input_dim,
        causal_nodes=config.model.causal_nodes,
        action_space=config.model.action_space
    )
    
    trainer = ACIETrainer(
        model=model,
        learning_rate=config.training.learning_rate,
        lambda_dag=config.training.lambda_dag,
        device=device
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs
    )
    
    # Generate causal graph
    print("\n[3/4] Generating causal graph visualization...")
    model.eval()
    generate_causal_graph(model, output_dir / "causal_graph.png", threshold=0.05)
    
    # Generate robustness plot
    print("\n[4/4] Evaluating adversarial robustness...")
    model.eval()
    epsilons, accuracies = evaluate_robustness(
        model, test_loader, device,
        epsilons=[0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    )
    generate_robustness_plot(epsilons, accuracies, output_dir / "robustness_plot.png")
    
    print("\n" + "=" * 60)
    print("Artifact generation complete.")
    print(f"Output files:")
    print(f"  - {output_dir / 'causal_graph.png'}")
    print(f"  - {output_dir / 'robustness_plot.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
