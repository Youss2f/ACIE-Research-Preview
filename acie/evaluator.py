import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging

logger = logging.getLogger(__name__)


class ACIEEvaluator:
    """
    Comprehensive evaluation suite for ACIE_Core including:
    - Classification metrics
    - Causal graph visualization
    - Robustness analysis
    - Adversarial testing
    """
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader) -> Dict:
        """
        Comprehensive evaluation on test set
        """
        all_preds = []
        all_targets = []
        all_probs = []
        no_threat_count = 0
        
        with torch.no_grad():
            for event_streams, targets in test_loader:
                event_streams = event_streams.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(event_streams)
                
                if outputs == "NO_THREAT":
                    no_threat_count += event_streams.size(0)
                    continue
                
                action_probs, _ = outputs
                predictions = torch.argmax(action_probs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(action_probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = {
            "accuracy": np.mean(all_preds == all_targets),
            "no_threat_filtered": no_threat_count,
            "total_samples": len(all_targets) + no_threat_count,
            "classification_report": classification_report(
                all_targets, all_preds, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(all_targets, all_preds)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Filtered (NO_THREAT): {no_threat_count}")
        logger.info(f"  Total Samples: {metrics['total_samples']}")
        
        return metrics, (all_preds, all_targets, all_probs)
    
    def visualize_causal_graph(self, save_path: str = None):
        """
        Visualize the learned causal adjacency matrix
        """
        adjacency = self.model.adjacency.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            adjacency,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Causal Weight"}
        )
        plt.title("Learned Causal Graph (Adjacency Matrix)")
        plt.xlabel("Effect Node")
        plt.ylabel("Cause Node")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Causal graph saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, confusion_mat, save_path: str = None):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar_kws={"label": "Count"}
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Action")
        plt.ylabel("True Action")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def adversarial_robustness_test(
        self,
        test_loader,
        epsilon_values: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3]
    ) -> Dict:
        """
        Test model robustness against adversarial perturbations
        """
        robustness_results = {}
        
        for epsilon in epsilon_values:
            correct = 0
            total = 0
            
            for event_streams, targets in test_loader:
                event_streams = event_streams.to(self.device)
                targets = targets.to(self.device)
                
                # Add adversarial noise
                noise = torch.randn_like(event_streams) * epsilon
                perturbed_streams = event_streams + noise
                
                outputs = self.model(perturbed_streams)
                
                if outputs == "NO_THREAT":
                    continue
                
                action_probs, _ = outputs
                predictions = torch.argmax(action_probs, dim=1)
                
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            robustness_results[epsilon] = accuracy
            logger.info(f"Robustness @ ε={epsilon:.2f}: {accuracy:.4f}")
        
        return robustness_results
    
    def plot_robustness_curve(self, robustness_results: Dict, save_path: str = None):
        """
        Plot robustness vs perturbation strength
        """
        epsilons = list(robustness_results.keys())
        accuracies = list(robustness_results.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel("Perturbation Strength (ε)")
        plt.ylabel("Accuracy")
        plt.title("Adversarial Robustness Analysis")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Robustness curve saved to {save_path}")
        
        plt.close()
    
    def analyze_information_filter(self, test_loader) -> Dict:
        """
        Analyze the information filter's entropy distribution
        """
        entropy_values = []
        filtered_count = 0
        passed_count = 0
        
        with torch.no_grad():
            for event_streams, _ in test_loader:
                event_streams = event_streams.to(self.device)
                
                for stream in event_streams:
                    hist = torch.histc(stream, bins=100)
                    prob = hist / torch.sum(hist)
                    h = -torch.sum(prob * torch.log(prob + 1e-9))
                    
                    entropy_values.append(h.item())
                    
                    if h < 3.5:
                        filtered_count += 1
                    else:
                        passed_count += 1
        
        analysis = {
            "mean_entropy": np.mean(entropy_values),
            "std_entropy": np.std(entropy_values),
            "min_entropy": np.min(entropy_values),
            "max_entropy": np.max(entropy_values),
            "filtered_ratio": filtered_count / (filtered_count + passed_count),
            "total_filtered": filtered_count,
            "total_passed": passed_count
        }
        
        logger.info("Information Filter Analysis:")
        logger.info(f"  Mean Entropy: {analysis['mean_entropy']:.4f}")
        logger.info(f"  Filtered: {filtered_count} ({analysis['filtered_ratio']*100:.2f}%)")
        logger.info(f"  Passed: {passed_count}")
        
        return analysis, entropy_values
    
    def plot_entropy_distribution(self, entropy_values: List[float], save_path: str = None):
        """
        Plot entropy distribution with threshold
        """
        plt.figure(figsize=(10, 6))
        plt.hist(entropy_values, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=3.5, color='r', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel("Shannon Entropy")
        plt.ylabel("Frequency")
        plt.title("Information Filter: Entropy Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entropy distribution saved to {save_path}")
        
        plt.close()
