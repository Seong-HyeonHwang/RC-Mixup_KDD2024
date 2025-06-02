import argparse
import numpy as np
import torch
from model import *
from utils import *
from train import *

def parse_args():
    parser = argparse.ArgumentParser(description='RC-Mixup: Data Augmentation for Noisy Regression')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='Synthetic_dataset.npy',
                       help='Path to dataset file')
    parser.add_argument('--label_dim', type=int, default=4,
                       help='Dimension of labels')
    parser.add_argument('--noise_ratio', type=float, default=0.3,
                       help='Ratio of noisy data (0.0-1.0)')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=410,
                       help='Random seed')
    parser.add_argument('--warmup_epochs', type=int, default=1000,
                       help='Number of warmup epochs')
    parser.add_argument('--update_epochs', type=int, default=1000,
                       help='Number of update epochs')
    parser.add_argument('--lr_model', type=float, default=1e-2,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    
    # RC-Mixup parameters
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='Alpha parameter for RC-Mixup')
    parser.add_argument('--L', type=int, default=500,
                       help='L parameter')
    parser.add_argument('--N', type=int, default=500,
                       help='N parameter')
    parser.add_argument('--bw_list', nargs='+', type=int, default=[5, 10, 15, 20],
                       help='Bandwidth list')
    parser.add_argument('--start_bandwidth', type=int, default=20,
                       help='Starting bandwidth')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()

def load_and_preprocess_data(args):
    """Load and preprocess the dataset"""
    print(f"Loading data from {args.data_path}...")
    raw_data = np.load(args.data_path)
    
    print(f"Preprocessing data with noise ratio: {args.noise_ratio}")
    train_data, valid_data, test_data, Y_max, Y_min, noise_idx = preprocess_dataset(
        args.seed, raw_data, args.noise_ratio
    )
    
    # Split features and labels
    x_tr, y_tr = train_data[:, :-args.label_dim], train_data[:, -args.label_dim:]
    x_val, y_val = valid_data[:, :-args.label_dim], valid_data[:, -args.label_dim:]
    x_test, y_test = test_data[:, :-args.label_dim], test_data[:, -args.label_dim:]
    
    # Create dataset dictionary
    dataset = {args.seed: {
        'x_tr': x_tr, 'y_tr': y_tr,
        'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test,
        'Y_max': Y_max, 'Y_min': Y_min,
        'noise_idx': noise_idx
    }}
    
    print(f"Dataset loaded: Train({x_tr.shape}), Valid({x_val.shape}), Test({x_test.shape})")
    return dataset

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and preprocess data
    dataset = load_and_preprocess_data(args)
    
    # Print hyperparameters
    print("\n" + "="*50)
    print("RC-Mixup Configuration:")
    print("="*50)
    print(f"Noise ratio: {args.noise_ratio}")
    print(f"Alpha: {args.alpha}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Update epochs: {args.update_epochs}")
    print(f"Learning rate: {args.lr_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Bandwidth list: {args.bw_list}")
    print("="*50 + "\n")
    
    # Run RC-Mixup algorithm
    print("Starting RC-Mixup training...")
    updated_model = rc_mixup(
        dataset=dataset,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        lr_model=args.lr_model,
        alpha=args.alpha,
        warmup_epochs=args.warmup_epochs,
        update_epochs=args.update_epochs,
        L=args.L,
        N=args.N,
        bw_list=args.bw_list,
        start_bandwidth=args.start_bandwidth,
        batch_size=args.batch_size,
        device=device
    )
    
    print("Training completed!")
    return updated_model

if __name__ == "__main__":
    model = main()