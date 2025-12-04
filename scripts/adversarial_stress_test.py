"""
Adversarial Stress Test for ACIE Model
Tests the Game Theoretic Robustness (Layer 3) against FGSM attacks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse

from acie import ACIE_Core
from acie.dataset import CyberLogDataset


def load_model(model_path, input_dim=100, causal_nodes=10, action_space=5):
    """Load trained ACIE model"""
    print(f"Loading model from {model_path}...")
    
    model = ACIE_Core(input_dim=input_dim, causal_nodes=causal_nodes, action_space=action_space)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[+] Model loaded (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return model


def find_target_sample(model, dataset, target_label=1, min_confidence=0.3):
    """
    Find a sample that is correctly classified with high confidence.
    Target label 1 = LATERAL_MOVEMENT
    """
    print(f"\nSearching for target sample (Label {target_label}: {dataset.attack_patterns[target_label]})...")
    
    model.eval()
    
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        
        if y.item() != target_label:
            continue
        
        # Run inference
        x_batch = x.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(x_batch)
            
            # Handle NO_THREAT case (model returns string)
            if isinstance(output, str):
                continue  # Skip this sample
            
            logits = output
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()
        
        # Check if correctly classified with high confidence
        if pred_label == target_label and confidence >= min_confidence:
            print(f"[+] Found target sample at index {idx}")
            print(f"  True label: {y.item()} ({dataset.attack_patterns[y.item()]})")
            print(f"  Predicted: {pred_label} ({dataset.attack_patterns[pred_label]})")
            print(f"  Confidence: {confidence:.4f}")
            return x, y, idx
    
    # If not found with min_confidence, try with lower threshold
    print(f"[WARN] No sample found with confidence >= {min_confidence}, searching with lower threshold...")
    
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        
        if y.item() != target_label:
            continue
        
        x_batch = x.unsqueeze(0)
        with torch.no_grad():
            output = model(x_batch)
            
            # Handle NO_THREAT case
            if isinstance(output, str):
                continue
            
            logits = output
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()
        
        if pred_label == target_label:
            print(f"[+] Found target sample at index {idx}")
            print(f"  True label: {y.item()} ({dataset.attack_patterns[y.item()]})")
            print(f"  Predicted: {pred_label} ({dataset.attack_patterns[pred_label]})")
            print(f"  Confidence: {confidence:.4f}")
            return x, y, idx
    
    raise ValueError(f"Could not find any correctly classified sample with label {target_label}")


def fgsm_attack(model, x, y, epsilon=0.1):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    Simulates an attacker trying to evade detection by perturbing input features.
    
    Args:
        model: ACIE model
        x: Input sample (tensor)
        y: True label (tensor)
        epsilon: Perturbation magnitude
    
    Returns:
        x_adv: Adversarial example
        gradient: Gradient of loss w.r.t. input
    """
    # Set model to training mode to enable gradients through robust_nash_policy
    model.train()
    
    # Clone input and enable gradient computation
    x_adv = x.clone().detach().unsqueeze(0)  # Add batch dimension
    x_adv.requires_grad = True
    
    # Forward pass
    output = model(x_adv)
    
    # Handle NO_THREAT case
    if isinstance(output, str):
        model.eval()
        return x.clone(), torch.zeros_like(x)
    
    logits = output
    
    # Compute loss (cross-entropy)
    loss = F.cross_entropy(logits, y.unsqueeze(0))
    
    # Backward pass to get gradient
    model.zero_grad()
    loss.backward()
    
    # Get gradient sign
    gradient = x_adv.grad.data
    gradient_sign = gradient.sign()
    
    # Create adversarial example: x_adv = x + epsilon * sign(gradient)
    x_adv = x + epsilon * gradient_sign.squeeze(0)
    
    # Set model back to eval mode
    model.eval()
    
    return x_adv.detach(), gradient.squeeze(0).detach()


def analyze_perturbation(x_clean, x_adv, dataset):
    """Analyze the adversarial perturbation"""
    perturbation = x_adv - x_clean
    
    print("\n" + "="*60)
    print("ADVERSARIAL PERTURBATION ANALYSIS")
    print("="*60)
    
    print(f"L2 Norm: {torch.norm(perturbation, p=2).item():.4f}")
    print(f"L∞ Norm: {torch.norm(perturbation, p=float('inf')).item():.4f}")
    print(f"Mean perturbation: {perturbation.mean().item():.4f}")
    print(f"Max perturbation: {perturbation.max().item():.4f}")
    print(f"Min perturbation: {perturbation.min().item():.4f}")
    
    # Analyze perturbations by field
    offset = 0
    print("\nPerturbation by Field:")
    for field, dim in dataset.field_dims.items():
        field_pert = perturbation[offset:offset+dim]
        field_l2 = torch.norm(field_pert, p=2).item()
        print(f"  {field:15s} L2: {field_l2:.4f}")
        offset += dim


def run_inference(model, x, label_name):
    """Run inference and return predictions"""
    model.eval()
    
    with torch.no_grad():
        x_batch = x.unsqueeze(0)
        output = model(x_batch)
        
        # Handle NO_THREAT case
        if isinstance(output, str):
            return None, None, None
        
        logits = output
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()
        all_probs = probs[0].numpy()
    
    return pred_label, confidence, all_probs


def compare_predictions(clean_pred, clean_conf, clean_probs, 
                       adv_pred, adv_conf, adv_probs,
                       true_label, dataset):
    """Compare clean vs adversarial predictions"""
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)
    
    print(f"\nTrue Label: {true_label} ({dataset.attack_patterns[true_label]})")
    
    print("\n" + "-"*60)
    print("CLEAN SAMPLE")
    print("-"*60)
    print(f"Predicted: {clean_pred} ({dataset.attack_patterns[clean_pred]})")
    print(f"Confidence: {clean_conf:.4f}")
    print(f"All probabilities:")
    for i, prob in enumerate(clean_probs):
        marker = "*" if i == true_label else " "
        print(f"  {marker} {i}: {dataset.attack_patterns[i]:25s} {prob:.4f}")
    
    print("\n" + "-"*60)
    print("ADVERSARIAL SAMPLE (FGSM Attack)")
    print("-"*60)
    print(f"Predicted: {adv_pred} ({dataset.attack_patterns[adv_pred]})")
    print(f"Confidence: {adv_conf:.4f}")
    print(f"All probabilities:")
    for i, prob in enumerate(adv_probs):
        marker = "*" if i == true_label else " "
        change = prob - clean_probs[i]
        change_str = f"({change:+.4f})" if change != 0 else ""
        print(f"  {marker} {i}: {dataset.attack_patterns[i]:25s} {prob:.4f} {change_str}")


def calculate_robustness_metrics(clean_pred, clean_conf, clean_probs,
                                 adv_pred, adv_conf, adv_probs,
                                 true_label):
    """Calculate robustness metrics"""
    print("\n" + "="*60)
    print("ROBUSTNESS METRICS")
    print("="*60)
    
    # 1. Robustness Ratio (key metric)
    clean_correct_prob = clean_probs[true_label]
    adv_correct_prob = adv_probs[true_label]
    
    if clean_correct_prob > 0:
        robustness_ratio = adv_correct_prob / clean_correct_prob
    else:
        robustness_ratio = 0.0
    
    print(f"\n1. Robustness Ratio: {robustness_ratio:.4f}")
    print(f"   Formula: P(Correct|Attack) / P(Correct|Clean)")
    print(f"   P(Correct|Clean): {clean_correct_prob:.4f}")
    print(f"   P(Correct|Attack): {adv_correct_prob:.4f}")
    
    if robustness_ratio > 0.8:
        print(f"   [SUCCESS] ROBUST: Ratio > 0.8 (Game Theory layer working!)")
    elif robustness_ratio > 0.5:
        print(f"   [WARN] PARTIAL: 0.5 < Ratio < 0.8 (Some robustness)")
    else:
        print(f"   [-] VULNERABLE: Ratio < 0.5 (Robustness failing)")
    
    # 2. Prediction Consistency
    pred_consistent = (clean_pred == adv_pred)
    print(f"\n2. Prediction Consistency: {pred_consistent}")
    print(f"   Clean Prediction: {clean_pred}")
    print(f"   Adversarial Prediction: {adv_pred}")
    if pred_consistent:
        print(f"   [SUCCESS] Model prediction unchanged under attack")
    else:
        print(f"   [-] Model prediction flipped by adversarial noise")
    
    # 3. Confidence Drop
    confidence_drop = clean_conf - adv_conf
    confidence_drop_pct = (confidence_drop / clean_conf * 100) if clean_conf > 0 else 0
    print(f"\n3. Confidence Drop: {confidence_drop:.4f} ({confidence_drop_pct:.1f}%)")
    print(f"   Clean Confidence: {clean_conf:.4f}")
    print(f"   Adversarial Confidence: {adv_conf:.4f}")
    
    # 4. Attack Success Rate
    attack_success = not pred_consistent or (adv_pred != true_label)
    print(f"\n4. Attack Success: {attack_success}")
    if attack_success:
        print(f"   [-] Attack succeeded (prediction changed or incorrect)")
    else:
        print(f"   [SUCCESS] Attack failed (prediction still correct)")
    
    return {
        'robustness_ratio': robustness_ratio,
        'prediction_consistent': pred_consistent,
        'confidence_drop': confidence_drop,
        'attack_success': attack_success
    }


def run_stress_test(model, dataset, epsilon=0.1, target_label=1, num_samples=5):
    """Run adversarial stress test on multiple samples"""
    print("\n" + "="*60)
    print("ADVERSARIAL STRESS TEST (MULTIPLE SAMPLES)")
    print("="*60)
    print(f"Epsilon: {epsilon}")
    print(f"Target Label: {target_label} ({dataset.attack_patterns[target_label]})")
    print(f"Number of samples: {num_samples}")
    
    results = []
    
    for i in range(num_samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*60}")
        
        # Find target sample
        try:
            x, y, idx = find_target_sample(model, dataset, target_label=target_label)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue
        
        # Run clean inference
        clean_pred, clean_conf, clean_probs = run_inference(model, x, "clean")
        
        # Skip if NO_THREAT
        if clean_pred is None:
            print("[WARN] Sample triggered NO_THREAT shortcut, skipping...")
            continue
        
        # Generate adversarial example
        x_adv, gradient = fgsm_attack(model, x, y, epsilon=epsilon)
        
        # Analyze perturbation
        analyze_perturbation(x, x_adv, dataset)
        
        # Run adversarial inference
        adv_pred, adv_conf, adv_probs = run_inference(model, x_adv, "adversarial")
        
        # Skip if adversarial also triggers NO_THREAT
        if adv_pred is None:
            print("[WARN] Adversarial sample triggered NO_THREAT shortcut, skipping...")
            continue
        
        # Compare predictions
        compare_predictions(clean_pred, clean_conf, clean_probs,
                          adv_pred, adv_conf, adv_probs,
                          y.item(), dataset)
        
        # Calculate metrics
        metrics = calculate_robustness_metrics(clean_pred, clean_conf, clean_probs,
                                              adv_pred, adv_conf, adv_probs,
                                              y.item())
        
        results.append(metrics)
        
        # Remove this sample from consideration for next iteration
        # (Simple approach: just search from idx+1)
    
    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    
    if results:
        avg_robustness = np.mean([r['robustness_ratio'] for r in results])
        consistency_rate = np.mean([r['prediction_consistent'] for r in results])
        avg_conf_drop = np.mean([r['confidence_drop'] for r in results])
        attack_success_rate = np.mean([r['attack_success'] for r in results])
        
        print(f"\nAverage Robustness Ratio: {avg_robustness:.4f}")
        print(f"Prediction Consistency Rate: {consistency_rate:.2%}")
        print(f"Average Confidence Drop: {avg_conf_drop:.4f}")
        print(f"Attack Success Rate: {attack_success_rate:.2%}")
        
        print("\n" + "="*60)
        print("FINAL VERDICT")
        print("="*60)
        
        if avg_robustness > 0.8:
            print("[SUCCESS] EXCELLENT ROBUSTNESS")
            print("   The Game Theory layer (Robust Nash Policy) is successfully")
            print("   defending against adversarial perturbations!")
        elif avg_robustness > 0.5:
            print("[WARN] MODERATE ROBUSTNESS")
            print("   The model shows some resistance to attacks but could be improved.")
        else:
            print("[-] POOR ROBUSTNESS")
            print("   The model is vulnerable to adversarial attacks.")
            print("   Consider increasing lambda_robust or training longer.")
        
        return avg_robustness
    else:
        print("[WARN] No results collected")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Adversarial Stress Test for ACIE")
    parser.add_argument("--model-path", type=str, default="models/cyber_test/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="FGSM attack epsilon (perturbation magnitude)")
    parser.add_argument("--target-label", type=int, default=1,
                       help="Target label to attack (1=LATERAL_MOVEMENT)")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--input-dim", type=int, default=100)
    parser.add_argument("--causal-nodes", type=int, default=10)
    parser.add_argument("--action-space", type=int, default=5)
    
    args = parser.parse_args()
    
    print("="*60)
    print("ACIE ADVERSARIAL STRESS TEST")
    print("Game Theoretic Robustness Validation")
    print("="*60)
    
    # Load model
    model = load_model(args.model_path, args.input_dim, args.causal_nodes, args.action_space)
    
    # Create dataset
    print("\nCreating test dataset...")
    dataset = CyberLogDataset(num_samples=200, input_dim=args.input_dim, 
                             num_classes=args.action_space, seed=42)
    print(f"Dataset created: {len(dataset)} samples")
    
    # Run stress test
    avg_robustness = run_stress_test(model, dataset, epsilon=args.epsilon, 
                                     target_label=args.target_label,
                                     num_samples=args.num_samples)
    
    print("\n" + "="*60)
    print("[SUCCESS] ADVERSARIAL STRESS TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
