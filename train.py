import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from model import CustomModel
from utils import evaluate, set_seed, make_sp, make_sp_denorm
import copy

def train(model, optimizer, x_tr, y_tr, batch_size,
          mixup_option=False, alpha=1.0, use_manifold=False, use_c_mixup=False, sp=None, device = None):
    """
    Trains the model for one epoch with optional Mixup augmentation.

    Args:
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        x_tr (np.ndarray): Training features.
        y_tr (np.ndarray): Training labels.
        batch_size (int): Size of each training batch.
        mixup_option (int, optional): Flag to enable Mixup. Defaults to False.
        alpha (float, optional): Mixup hyperparameter. Defaults to 1.0.
        use_manifold (bool, optional): Flag to use manifold Mixup. Defaults to False.
        use_c_mixup (bool, optional): Flag to use custom Mixup. Defaults to False.
        sp (np.ndarray, optional): Probability matrix for C-Mixup. Defaults to None.

    Returns:
        nn.Module: The trained model.
    """
    model.train()
 
    dataset_size = len(y_tr)
    
    # Calculate number of iterations
    iterations = max(dataset_size // batch_size, 1)
    
    # Shuffle indices
    shuffle_idx = torch.randperm(dataset_size)
    
    mse_loss = nn.MSELoss(reduction='mean')
    
    for i in range(iterations):
        start = i * batch_size
        end = start + batch_size
        batch_indices = shuffle_idx[start:end]
        
        data_1 = torch.tensor(x_tr[batch_indices], dtype=torch.float32).to(device)
        label_1 = torch.tensor(y_tr[batch_indices], dtype=torch.float32).to(device)
        
        if mixup_option:
            lambd = np.random.beta(alpha, alpha)
            
            if use_c_mixup and sp is not None:
                # Sample mixed indices based on probability matrix 'sp'
                mixed_indices = np.array([np.random.choice(y_tr.shape[0], p = sp[idx]) for idx in batch_indices])
            else:
                # Sample mixed indices from mixup_idx_dict
                mixed_indices = np.array([np.random.choice(y_tr.shape[0]) for idx in batch_indices])
            
            data_2 = torch.tensor(x_tr[mixed_indices], dtype=torch.float32).to(device)
            label_2 = torch.tensor(y_tr[mixed_indices], dtype=torch.float32).to(device)
            
            if use_manifold:
                pred_labels = model.forward_mixup(data_1, data_2, lambd)
            else:
                mixed_data = lambd * data_1 + (1 - lambd) * data_2
                pred_labels = model(mixed_data)
            
            mixed_labels = lambd * label_1 + (1 - lambd) * label_2
            loss = mse_loss(pred_labels, mixed_labels)
        else:
            pred_labels = model(data_1)
            loss = mse_loss(pred_labels, label_1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def itlm_train(model, optimizer, batch_size, alpha, x_tr, y_tr, x_val, y_val,
              sp_denorm, num_noise, device=None):
    """
    Iteratively trains the model while identifying and handling noisy labels.

    Args:
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        batch_size (int): Size of each training batch.
        alpha (float): Mixup hyperparameter.
        x_tr (np.ndarray): Training features.
        y_tr (np.ndarray): Training labels.
        x_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        sp_denorm (np.ndarray): Denormalized probability matrix for C-Mixup.
        num_noise (int): Number of noisy samples to handle.

    Returns:
        tuple: Updated model, validation RMSE
    """
    
    model.train()
    mse_loss = nn.MSELoss(reduction='none')
    
    # Convert training data to tensors
    x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_out = model(x_tr_tensor)
        losses = torch.mean(mse_loss(y_out, y_tr_tensor), dim=1)
    
    # Sort indices by loss in ascending order
    sorted_indices = torch.argsort(losses).cpu().numpy()
    clean_idx = sorted_indices[:-num_noise]
    
    # Select clean data
    x_tr_clean = x_tr[clean_idx]
    y_tr_clean = y_tr[clean_idx]
    
    # Normalize the probability matrix
    sp = sp_denorm[clean_idx][:, clean_idx]
    sp = sp/np.sum(sp, axis = -1).reshape(-1, 1).repeat(len(clean_idx), axis = 1)
    
    # Train the model on clean data
    model = train(model, optimizer, x_tr_clean, y_tr_clean, batch_size=batch_size, 
                 mixup_option=True, alpha=alpha, use_manifold=False, use_c_mixup=True, sp=sp, device = device)
    
    # Evaluate on validation and test sets
    with torch.no_grad():
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        pred_y_val = model(x_val_tensor).cpu().numpy()
        val_rmse = mean_squared_error(y_val, pred_y_val) ** 0.5
    
    return model, val_rmse

def rc_mixup(dataset, noise_ratio, seed, lr_model, alpha, warmup_epochs, update_epochs,
             L, N, bw_list, start_bandwidth, batch_size, device=None):
    """
    Main function to perform RC-Mixup training across multiple seeds.

    Args:
        dataset (dict): Dataset containing training, validation, and test splits.
        noise_ratio (float): Ratio of noisy samples.
        seed (int): A random seed.
        lr_model (float): Learning rate for the optimizer.
        alpha (float): Mixup hyperparameter.
        warmup_epochs (int): Number of training epochs for warm-up phase.
        update_epochs (int): Number of training epochs for clean and update phase.
        L, N (int): Hyperparameters for dynamic bandwidth tuning.
        bw_list (list) : A set of possible bandwidths.
        start_bandwidth (float): A initial bandwidth before bandwidth tuning.
        batch_size (int): Size of each training batch.

    Returns:
        model
    """
    start_time = time.time()
    
    # Extract dataset splits
    split = dataset[seed]
    x_tr, y_tr = split['x_tr'], split['y_tr']
    x_val, y_val = split['x_val'], split['y_val']
    x_test, y_test = split['x_test'], split['y_test']
    n_noise = int(noise_ratio * len(x_tr))
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Initialize model and optimizer
    model = CustomModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_model)
    
    best_rmse_overall = float('inf')
    sp = make_sp(y_tr, start_bandwidth)
    
    # Warm-up phase
    for epoch in range(warmup_epochs):
        model = train(model, optimizer, x_tr, y_tr, batch_size=batch_size, 
                        mixup_option=True, alpha=alpha, use_manifold=False, use_c_mixup=True, sp=sp, device=device)
        
        # Evaluate on validation
        val_rmse, _ = evaluate(model, x_val, y_val, device)
        
        # Save the best model based on validation RMSE
        if val_rmse < best_rmse_overall:
            best_rmse_overall = val_rmse
            best_model = copy.deepcopy(model)
    
    # Load the best model
    model = copy.deepcopy(best_model)
    optimizer = optim.Adam(model.parameters(), lr=lr_model)  # Reinitialize optimizer
    best_rmse_overall = float('inf')
        
    epoch = 0
    sp_denorm = make_sp_denorm(y_tr, start_bandwidth)
    
    # Update phase
    while epoch < update_epochs:
        if epoch % L == 0:
            # Generate denormalized probability matrices for different bandwidths
            sp_denorm_list = [make_sp_denorm(y_tr, bw=bw) for bw in bw_list]
            
            best_rmse_during_search = float('inf')
            best_model_search = None
            
            for bw_idx, bw in enumerate(bw_list):
                # Initialize a new model for each bandwidth
                test_model = copy.deepcopy(model)
                test_optimizer = optim.Adam(test_model.parameters(), lr=lr_model)
                test_optimizer.load_state_dict(optimizer.state_dict())
                
                sp_denorm_bw = sp_denorm_list[bw_idx]
                best_rmse_per_bw = float('inf')
                
                for _ in range(N):
                    # Train iteratively with ITLM
                    test_model, val_rmse = itlm_train(test_model, test_optimizer, batch_size, alpha,
                        x_tr, y_tr, x_val, y_val,
                        sp_denorm_bw, n_noise, device=device)
                    
                    # Update best RMSE and model state if improved
                    if val_rmse <= best_rmse_per_bw:
                        best_rmse_per_bw = val_rmse
                        best_model_search = copy.deepcopy(test_model)
                        
                if best_rmse_per_bw <= best_rmse_during_search:
                    best_rmse_during_search = best_rmse_per_bw
                    next_model = copy.deepcopy(test_model)
                    next_model_optimizer = optim.Adam(next_model.parameters(), lr = lr_model)
                    next_model_optimizer.load_state_dict(test_optimizer.state_dict())
                    current_best_bw_idx = bw_idx
                    
                    if best_rmse_during_search <= best_rmse_overall:
                        best_model = copy.deepcopy(best_model_search)
                        best_rmse_overall = best_rmse_during_search
        
            model = copy.deepcopy(next_model)
            optimizer = optim.Adam(model.parameters(), lr=lr_model)  
            optimizer.load_state_dict(next_model_optimizer.state_dict())
            epoch += N
            sp_denorm = sp_denorm_list[current_best_bw_idx]
        else:
            # Train iteratively without bandwidth tuning
            model, curr_val_rmse = itlm_train(model, optimizer, batch_size, alpha, x_tr, y_tr, x_val, y_val,
                sp_denorm, n_noise, device=device)
            epoch += 1
            
            # Update best model if validation RMSE improves
            if curr_val_rmse < best_rmse_overall:
                best_rmse_overall = curr_val_rmse
                best_model = copy.deepcopy(model)
    
    # Final evaluation
    best_val_rmse, best_val_mape = evaluate(best_model, x_val, y_val, device)
    best_test_rmse, best_test_mape = evaluate(best_model, x_test, y_test, device)
    
    # Logging results
    print(f'Seed: {seed},\n'
          f'Val RMSE: {best_val_rmse:.4f}, Val MAPE: {best_val_mape:.4f},\n'
          f'Test RMSE: {best_test_rmse:.4f}, Test MAPE: {best_test_mape:.4f},\n'
          f'Runtime: {time.time() - start_time:.1f}s')

    return best_model