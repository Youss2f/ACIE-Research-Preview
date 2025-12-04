"""
Visualize the learned causal structure (adjacency matrix) from trained ACIE model.
This script extracts and plots the DAG learned during training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from acie import ACIE_Core


def load_model(model_path, input_dim=100, causal_nodes=10, action_space=5):
    """Load trained ACIE model"""
    print(f"Loading model from {model_path}...")
    
    model = ACIE_Core(input_dim=input_dim, causal_nodes=causal_nodes, action_space=action_space)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[+] Model loaded (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return model


def extract_adjacency(model, threshold=0.1):
    """Extract and threshold the adjacency matrix"""
    adjacency = model.adjacency.detach().cpu().numpy()
    
    print(f"\nAdjacency Matrix Statistics:")
    print(f"  Shape: {adjacency.shape}")
    print(f"  Min: {adjacency.min():.4f}, Max: {adjacency.max():.4f}")
    print(f"  Mean: {adjacency.mean():.4f}, Std: {adjacency.std():.4f}")
    
    # Apply threshold to remove weak edges
    adjacency_thresholded = adjacency.copy()
    adjacency_thresholded[np.abs(adjacency_thresholded) < threshold] = 0
    
    num_edges = np.sum(np.abs(adjacency_thresholded) > 0)
    print(f"  Edges after threshold ({threshold}): {num_edges}")
    
    return adjacency, adjacency_thresholded


def find_strongest_path(adjacency, max_length=5):
    """Find the strongest connected path (attack path)"""
    n = adjacency.shape[0]
    
    # Create graph with edge weights
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] != 0:
                G.add_edge(i, j, weight=abs(adjacency[i, j]))
    
    if len(G.edges()) == 0:
        return []
    
    # Find the strongest simple path by trying all node pairs
    strongest_path = []
    max_path_weight = 0
    
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                try:
                    # Get all simple paths between source and target
                    for path in nx.all_simple_paths(G, source, target, cutoff=max_length):
                        # Calculate path weight (sum of edge weights)
                        path_weight = sum(G[path[i]][path[i+1]]['weight'] 
                                        for i in range(len(path)-1))
                        
                        if path_weight > max_path_weight:
                            max_path_weight = path_weight
                            strongest_path = path
                except nx.NetworkXNoPath:
                    continue
    
    if strongest_path:
        print(f"\nStrongest Attack Path (weight: {max_path_weight:.4f}):")
        print(f"  {' -> '.join([f'Event {node}' for node in strongest_path])}")
    
    return strongest_path


def visualize_causal_graph(adjacency, strongest_path=None, save_path=None):
    """Create a visualization of the causal graph"""
    n = adjacency.shape[0]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add all nodes
    for i in range(n):
        G.add_node(i, label=f"Event {i}")
    
    # Add edges with weights
    edge_weights = []
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] != 0:
                G.add_edge(i, j, weight=adjacency[i, j])
                edge_weights.append(abs(adjacency[i, j]))
    
    if len(edge_weights) == 0:
        print("\n[WARN] No edges found after thresholding. Try lower threshold.")
        edge_weights = [1.0]  # Default for empty graph
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ===== Left Plot: Full Causal Graph =====
    ax1.set_title("Learned Causal Structure (DAG)", fontsize=14, fontweight='bold')
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if strongest_path and node in strongest_path:
            node_colors.append('#FF6B6B')  # Red for attack path
            node_sizes.append(1200)
        else:
            node_colors.append('#4ECDC4')  # Teal for other nodes
            node_sizes.append(800)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          ax=ax1, alpha=0.9)
    
    # Draw edges
    regular_edges = []
    attack_edges = []
    regular_weights = []
    attack_weights = []
    
    for u, v in G.edges():
        weight = abs(adjacency[u, v])
        if strongest_path and len(strongest_path) > 1:
            # Check if edge is in attack path
            is_attack_edge = any(strongest_path[i] == u and strongest_path[i+1] == v 
                                for i in range(len(strongest_path)-1))
            if is_attack_edge:
                attack_edges.append((u, v))
                attack_weights.append(weight)
            else:
                regular_edges.append((u, v))
                regular_weights.append(weight)
        else:
            regular_edges.append((u, v))
            regular_weights.append(weight)
    
    # Normalize edge widths
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    weight_range = max_weight - min_weight if max_weight > min_weight else 1
    
    # Draw regular edges
    if regular_edges:
        edge_widths = [1 + 4 * (w - min_weight) / weight_range for w in regular_weights]
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=edge_widths,
                             edge_color='#95A5A6', alpha=0.6, ax=ax1, 
                             arrowsize=15, arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Draw attack path edges (highlighted)
    if attack_edges:
        attack_edge_widths = [2 + 6 * (w - min_weight) / weight_range for w in attack_weights]
        nx.draw_networkx_edges(G, pos, edgelist=attack_edges, width=attack_edge_widths,
                             edge_color='#FF6B6B', alpha=0.9, ax=ax1, 
                             arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    labels = {i: f"E{i}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', 
                           font_color='white', ax=ax1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Attack Path Nodes'),
        Patch(facecolor='#4ECDC4', label='Regular Nodes'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax1.axis('off')
    ax1.set_xlim([x - 0.1 for x in ax1.get_xlim()])
    ax1.set_ylim([y - 0.1 for y in ax1.get_ylim()])
    
    # ===== Right Plot: Adjacency Matrix Heatmap =====
    ax2.set_title("Adjacency Matrix Heatmap", fontsize=14, fontweight='bold')
    
    im = ax2.imshow(adjacency, cmap='RdBu_r', aspect='auto', vmin=-max_weight, vmax=max_weight)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Causal Weight', rotation=270, labelpad=20, fontsize=10)
    
    # Add grid
    ax2.set_xticks(np.arange(n))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels([f'E{i}' for i in range(n)], fontsize=9)
    ax2.set_yticklabels([f'E{i}' for i in range(n)], fontsize=9)
    ax2.set_xlabel('Effect (To)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cause (From)', fontsize=11, fontweight='bold')
    
    # Add grid lines
    ax2.set_xticks(np.arange(n)-.5, minor=True)
    ax2.set_yticks(np.arange(n)-.5, minor=True)
    ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Annotate strongest connections
    for i in range(n):
        for j in range(n):
            if abs(adjacency[i, j]) > 0.2:  # Only show strong connections
                text_color = 'white' if abs(adjacency[i, j]) > max_weight/2 else 'black'
                ax2.text(j, i, f'{adjacency[i, j]:.2f}', 
                        ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n[+] Causal graph saved to: {save_path}")
    
    return fig


def print_causal_summary(adjacency, threshold=0.1):
    """Print summary of learned causal relationships"""
    print("\n" + "="*60)
    print("CAUSAL STRUCTURE ANALYSIS")
    print("="*60)
    
    # Find top causal relationships
    n = adjacency.shape[0]
    relationships = []
    for i in range(n):
        for j in range(n):
            if abs(adjacency[i, j]) >= threshold:
                relationships.append((i, j, adjacency[i, j]))
    
    # Sort by absolute weight
    relationships.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\nTop 10 Strongest Causal Relationships:")
    print("-" * 60)
    for idx, (i, j, weight) in enumerate(relationships[:10], 1):
        direction = "->" if weight > 0 else "-|"
        print(f"  {idx:2d}. Event {i} {direction} Event {j}  |  Weight: {weight:+.4f}")
    
    if len(relationships) == 0:
        print("  (No relationships above threshold)")
    
    # Network statistics
    print(f"\nNetwork Statistics:")
    print(f"  Total Nodes: {n}")
    print(f"  Total Edges: {len(relationships)}")
    print(f"  Density: {len(relationships)/(n*(n-1)):.2%}" if n > 1 else "  Density: N/A")
    print(f"  Sparsity: {1 - len(relationships)/(n*(n-1)):.2%}" if n > 1 else "  Sparsity: N/A")
    
    # Find nodes with most outgoing/incoming edges
    out_degree = {i: 0 for i in range(n)}
    in_degree = {i: 0 for i in range(n)}
    
    for i, j, _ in relationships:
        out_degree[i] += 1
        in_degree[j] += 1
    
    max_out_node = max(out_degree, key=out_degree.get)
    max_in_node = max(in_degree, key=in_degree.get)
    
    print(f"\n  Most Influential Node (outgoing): Event {max_out_node} ({out_degree[max_out_node]} edges)")
    print(f"  Most Affected Node (incoming): Event {max_in_node} ({in_degree[max_in_node]} edges)")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize ACIE Causal Graph")
    parser.add_argument("--model-path", type=str, default="models/hello_world/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Threshold for edge filtering")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for visualization")
    parser.add_argument("--input-dim", type=int, default=100)
    parser.add_argument("--causal-nodes", type=int, default=10)
    parser.add_argument("--action-space", type=int, default=5)
    
    args = parser.parse_args()
    
    print("="*60)
    print("ACIE Causal Structure Visualization")
    print("="*60)
    
    # Determine output path
    if args.output is None:
        model_dir = Path(args.model_path).parent
        args.output = model_dir / "causal_structure.png"
    
    # Load model
    model = load_model(args.model_path, args.input_dim, args.causal_nodes, args.action_space)
    
    # Extract adjacency matrix
    adjacency_full, adjacency_filtered = extract_adjacency(model, threshold=args.threshold)
    
    # Find strongest path
    strongest_path = find_strongest_path(adjacency_filtered, max_length=5)
    
    # Print summary
    print_causal_summary(adjacency_filtered, threshold=args.threshold)
    
    # Visualize
    fig = visualize_causal_graph(adjacency_filtered, strongest_path, save_path=args.output)
    
    print("\n[SUCCESS] Visualization complete")
    print(f"   View the causal graph at: {args.output}")
    

if __name__ == "__main__":
    main()
