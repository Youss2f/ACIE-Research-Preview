"""
Inference Script for ACIE Model
Performs real-time threat detection on incoming event streams
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from pathlib import Path

from acie import ACIE_Core
from acie.utils import setup_logging, get_device

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="ACIE Inference")
    
    # Model parameters
    parser.add_argument("--input-dim", type=int, default=100, help="Input dimension")
    parser.add_argument("--causal-nodes", type=int, default=10, help="Number of causal nodes")
    parser.add_argument("--action-space", type=int, default=5, help="Action space size")
    
    # Model loading
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    
    # Input
    parser.add_argument("--input-file", type=str, help="Path to input event stream file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # System
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    return parser.parse_args()


def load_model(model_path, input_dim, causal_nodes, action_space, device):
    """Load trained model"""
    model = ACIE_Core(
        input_dim=input_dim,
        causal_nodes=causal_nodes,
        action_space=action_space
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    return model


def predict(model, event_stream, device):
    """Perform inference on event stream"""
    # Convert to tensor
    if isinstance(event_stream, np.ndarray):
        event_stream = torch.from_numpy(event_stream).float()
    
    event_stream = event_stream.to(device)
    
    with torch.no_grad():
        output = model(event_stream)
    
    if output == "NO_THREAT":
        return {
            "threat_detected": False,
            "action": "NO_ACTION",
            "confidence": 1.0,
            "message": "Low entropy - benign traffic"
        }
    
    action_probs, adjacency = output
    action_idx = torch.argmax(action_probs, dim=1).cpu().item()
    confidence = action_probs[0, action_idx].cpu().item()
    
    return {
        "threat_detected": True,
        "action": f"ACTION_{action_idx}",
        "confidence": confidence,
        "action_probabilities": action_probs.cpu().numpy().tolist(),
        "causal_graph": adjacency.cpu().numpy().tolist()
    }


def interactive_mode(model, input_dim, device):
    """Run model in interactive mode"""
    logger.info("\n" + "=" * 50)
    logger.info("Interactive Mode - Enter 'quit' to exit")
    logger.info("=" * 50)
    
    while True:
        try:
            user_input = input("\nEnter event stream (comma-separated values) or 'random': ")
            
            if user_input.lower() == 'quit':
                logger.info("Exiting...")
                break
            
            if user_input.lower() == 'random':
                # Generate random event stream
                event_stream = np.random.randn(1, input_dim)
                logger.info("Generated random event stream")
            else:
                # Parse user input
                values = [float(x.strip()) for x in user_input.split(',')]
                if len(values) != input_dim:
                    logger.error(f"Expected {input_dim} values, got {len(values)}")
                    continue
                event_stream = np.array([values])
            
            # Predict
            result = predict(model, event_stream, device)
            
            # Display results
            print("\n" + "-" * 50)
            print("PREDICTION RESULTS")
            print("-" * 50)
            print(f"Threat Detected: {result['threat_detected']}")
            print(f"Recommended Action: {result['action']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            if result['threat_detected']:
                print("\nAction Probabilities:")
                for i, prob in enumerate(result['action_probabilities'][0]):
                    print(f"  ACTION_{i}: {prob:.4f}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def batch_inference(model, input_file, device):
    """Run batch inference on file"""
    logger.info(f"Loading data from {input_file}")
    
    # Load data (assuming CSV or NPY format)
    if input_file.endswith('.npy'):
        data = np.load(input_file)
    elif input_file.endswith('.csv'):
        import pandas as pd
        data = pd.read_csv(input_file).values
    else:
        raise ValueError("Unsupported file format. Use .npy or .csv")
    
    logger.info(f"Loaded {len(data)} event streams")
    
    results = []
    for i, event_stream in enumerate(data):
        result = predict(model, event_stream, device)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} streams")
    
    # Summary statistics
    threat_count = sum(1 for r in results if r['threat_detected'])
    logger.info(f"\nSummary:")
    logger.info(f"  Total streams: {len(results)}")
    logger.info(f"  Threats detected: {threat_count}")
    logger.info(f"  Benign traffic: {len(results) - threat_count}")
    
    return results


def main():
    args = parse_args()
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logger.info("=" * 50)
    logger.info("ACIE Inference Engine")
    logger.info("=" * 50)
    
    # Load model
    model = load_model(
        args.model_path,
        args.input_dim,
        args.causal_nodes,
        args.action_space,
        device
    )
    
    # Run inference
    if args.interactive:
        interactive_mode(model, args.input_dim, device)
    elif args.input_file:
        batch_inference(model, args.input_file, device)
    else:
        logger.error("Please specify either --interactive or --input-file")


if __name__ == "__main__":
    main()
