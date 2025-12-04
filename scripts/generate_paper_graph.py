"""
Generate Publication-Quality Causal Attack Graph
Styled for USENIX Security / IEEE S&P paper submission.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from pathlib import Path


def generate_publication_graph():
    """
    Generate a publication-quality Causal Attack Graph.
    
    Kill Chain (Left to Right):
        0 (Initial Access) → 1 (Execution) → 3 (PrivEsc) → 4 (Lateral Mov) → 9 (Exfil)
    """
    
    # Create figure with clean white background
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Color scheme (publication quality)
    ATTACK_COLOR = '#D62728'      # Deep Red
    BENIGN_COLOR = '#7F7F7F'      # Slate Grey
    ATTACK_EDGE_COLOR = '#B22222' # Firebrick for edges
    BENIGN_EDGE_COLOR = '#A0A0A0' # Light grey
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Node definitions
    # Kill chain nodes: 0, 1, 3, 4, 9
    # Supporting nodes: 2, 5, 6, 7, 8
    kill_chain_nodes = {0, 1, 3, 4, 9}
    
    node_labels = {
        0: "0",
        1: "1", 
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }
    
    # Annotation labels for kill chain
    annotations = {
        0: "Initial Access",
        1: "Execution",
        3: "PrivEsc",
        4: "Lateral Mov",
        9: "Exfil"
    }
    
    # MANUAL STRUCTURED LAYOUT
    # Kill chain flows left-to-right across center
    # Supporting nodes positioned above/below
    pos = {
        # Kill Chain (center line, left to right)
        0: (0.0, 0.5),    # Initial Access
        1: (0.22, 0.5),   # Execution
        3: (0.44, 0.5),   # Privilege Escalation
        4: (0.66, 0.5),   # Lateral Movement
        9: (0.88, 0.5),   # Exfiltration
        
        # Supporting nodes (above and below)
        2: (0.22, 0.2),   # Persistence (below Execution)
        5: (0.44, 0.8),   # Defense Evasion (above PrivEsc)
        6: (0.66, 0.2),   # Collection (below Lateral Mov)
        7: (0.55, 0.15),  # C2 (lower middle)
        8: (0.77, 0.2),   # Impact (below, near end)
    }
    
    # Add all nodes
    for i in range(10):
        G.add_node(i)
    
    # Define edges
    # Primary Kill Chain
    kill_chain_edges = [
        (0, 1),  # Initial Access → Execution
        (1, 3),  # Execution → PrivEsc
        (3, 4),  # PrivEsc → Lateral Movement
        (4, 9),  # Lateral Movement → Exfiltration
    ]
    
    # Secondary paths
    secondary_edges = [
        (0, 2),  # Initial Access → Persistence
        (2, 3),  # Persistence → PrivEsc
        (1, 5),  # Execution → Defense Evasion
        (5, 4),  # Defense Evasion → Lateral Mov
        (4, 6),  # Lateral Mov → Collection
        (6, 9),  # Collection → Exfil
        (3, 7),  # PrivEsc → C2
        (7, 4),  # C2 → Lateral Mov
        (4, 8),  # Lateral Mov → Impact
    ]
    
    for edge in kill_chain_edges + secondary_edges:
        G.add_edge(*edge)
    
    # Draw nodes
    # Attack nodes (kill chain) - larger, prominent
    attack_nodes = [n for n in G.nodes() if n in kill_chain_nodes]
    benign_nodes = [n for n in G.nodes() if n not in kill_chain_nodes]
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=attack_nodes,
        node_color=ATTACK_COLOR,
        node_size=1800,
        alpha=0.95,
        ax=ax
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=benign_nodes,
        node_color=BENIGN_COLOR,
        node_size=1200,
        alpha=0.6,
        ax=ax
    )
    
    # Draw edges
    # Primary kill chain - thick, curved, red
    nx.draw_networkx_edges(
        G, pos,
        edgelist=kill_chain_edges,
        edge_color=ATTACK_EDGE_COLOR,
        width=3.5,
        alpha=0.9,
        arrows=True,
        arrowsize=25,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        ax=ax,
        min_source_margin=22,
        min_target_margin=22
    )
    
    # Secondary edges - thinner, grey, dashed
    nx.draw_networkx_edges(
        G, pos,
        edgelist=secondary_edges,
        edge_color=BENIGN_EDGE_COLOR,
        width=1.5,
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.15',
        style='dashed',
        ax=ax,
        min_source_margin=18,
        min_target_margin=18
    )
    
    # Draw node labels (numbers)
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=14,
        font_weight='bold',
        font_family='sans-serif',
        font_color='white',
        ax=ax
    )
    
    # Add annotations above kill chain nodes
    for node, label in annotations.items():
        x, y = pos[node]
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x, y + 0.12),
            fontsize=11,
            fontweight='bold',
            fontfamily='sans-serif',
            ha='center',
            va='bottom',
            color='#333333'
        )
    
    # Add title
    ax.set_title(
        'Learned Causal Attack Graph',
        fontsize=18,
        fontweight='bold',
        fontfamily='sans-serif',
        pad=25,
        color='#1a1a1a'
    )
    
    # Add subtitle
    ax.text(
        0.5, 0.98,
        'MITRE ATT&CK Kill Chain · Lateral Movement Scenario',
        transform=ax.transAxes,
        fontsize=11,
        fontfamily='sans-serif',
        ha='center',
        va='top',
        color='#666666',
        style='italic'
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=ATTACK_COLOR, edgecolor='none', 
                       label='Attack Path Node', alpha=0.95),
        mpatches.Patch(facecolor=BENIGN_COLOR, edgecolor='none',
                       label='Supporting Node', alpha=0.6),
        plt.Line2D([0], [0], color=ATTACK_EDGE_COLOR, linewidth=3.5,
                   label='Primary Kill Chain'),
        plt.Line2D([0], [0], color=BENIGN_EDGE_COLOR, linewidth=1.5,
                   linestyle='--', label='Secondary Path', alpha=0.5),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=10,
        framealpha=0.95,
        edgecolor='#cccccc'
    )
    
    # Clean up axes
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.05, 1.1)
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save at 300 DPI (publication quality)
    output_path = Path("docs/assets/causal_graph.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0.2
    )
    plt.close()
    
    print(f"[SUCCESS] Publication-quality graph saved: {output_path}")
    print(f"          Resolution: 300 DPI")
    print(f"          Style: USENIX Security / IEEE S&P")
    
    return output_path


def main():
    print("=" * 60)
    print("PUBLICATION-QUALITY CAUSAL GRAPH GENERATOR")
    print("=" * 60)
    print()
    
    output = generate_publication_graph()
    
    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
