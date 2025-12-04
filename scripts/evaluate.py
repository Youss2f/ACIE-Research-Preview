"""
Evaluation Script for ACIE Model
Evaluates the trained model on test data and generates visualizations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
import argparse
from pathlib import Path

from acie import ACIE_Core, ACIEEvaluator
from acie.dataset import CyberEventDataset
from acie.utils import setup_logging, set_seed, get_device

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACIE Model")
    
    # Model parameters
    parser.add_argument("--input-dim", type=int, default=100, help="Input dimension")
    parser.add_argument("--causal-nodes", type=int, default=10, help="Number of causal nodes")
    parser.add_argument("--action-space", type=int, default=5, help="Action space size")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split ratio")
    
    # Model loading
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Output
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--experiment-name", type=str, default="eval", help="Experiment name")
    parser.add_argument("--adversarial-test", action="store_true", help="Run adversarial tests")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logger.info("=" * 50)
    logger.info("ACIE Evaluation Pipeline")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_path}")
    
    # Create results directory
    results_dir = Path(args.results_dir) / args.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    
    # ==================== Load Model ====================
    logger.info("\n" + "=" * 50)
    logger.info("Loading Model")
    logger.info("=" * 50)
    
    model = ACIE_Core(
        input_dim=args.input_dim,
        causal_nodes=args.causal_nodes,
        action_space=args.action_space
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    logger.info(f"Model loaded from {args.model_path}")
    
    # ==================== Load Test Data ====================
    logger.info("\n" + "=" * 50)
    logger.info("Loading Test Dataset")
    logger.info("=" * 50)
    
    dataset = CyberEventDataset(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        num_classes=args.action_space,
        seed=args.seed
    )
    
    # Use last portion as test set
    train_val_size = int((1 - args.test_split) * len(dataset))
    test_size = len(dataset) - train_val_size
    _, test_dataset = random_split(dataset, [train_val_size, test_size])
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # ==================== Evaluation ====================
    logger.info("\n" + "=" * 50)
    logger.info("Running Evaluation")
    logger.info("=" * 50)
    
    evaluator = ACIEEvaluator(model, device=device)
    
    # Standard evaluation
    metrics, (preds, targets, probs) = evaluator.evaluate(test_loader)
    
    # Save metrics
    import json
    with open(results_dir / "metrics.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {
            k: v.tolist() if hasattr(v, 'tolist') else v 
            for k, v in metrics.items() 
            if k != 'classification_report'
        }
        serializable_metrics['classification_report'] = metrics['classification_report']
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {results_dir / 'metrics.json'}")
    
    # ==================== Visualizations ====================
    logger.info("\n" + "=" * 50)
    logger.info("Generating Visualizations")
    logger.info("=" * 50)
    
    # Causal graph
    evaluator.visualize_causal_graph(save_path=results_dir / "causal_graph.png")
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=results_dir / "confusion_matrix.png"
    )
    
    # Information filter analysis
    filter_analysis, entropy_values = evaluator.analyze_information_filter(test_loader)
    evaluator.plot_entropy_distribution(
        entropy_values,
        save_path=results_dir / "entropy_distribution.png"
    )
    
    # ==================== Adversarial Testing ====================
    if args.adversarial_test:
        logger.info("\n" + "=" * 50)
        logger.info("Running Adversarial Robustness Tests")
        logger.info("=" * 50)
        
        robustness_results = evaluator.adversarial_robustness_test(
            test_loader,
            epsilon_values=[0.0, 0.05, 0.1, 0.2, 0.3]
        )
        
        evaluator.plot_robustness_curve(
            robustness_results,
            save_path=results_dir / "robustness_curve.png"
        )
        
        # Save robustness results
        with open(results_dir / "robustness_results.json", "w") as f:
            json.dump(robustness_results, f, indent=4)
    
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Complete!")
    logger.info(f"All results saved to {results_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
