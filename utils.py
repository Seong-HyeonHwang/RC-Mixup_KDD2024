from sklearn.metrics import mean_squared_error  
from sklearn.metrics import mean_absolute_percentage_error
import torch
import numpy as np
import random
from sklearn.neighbors import KernelDensity

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate(model, x, y, device) :
    model.eval()
    x_torch = torch.tensor(x).float().to(device)
    with torch.no_grad():
        pred_y = model(x_torch).cpu().data.numpy()
        rmse = mean_squared_error(y, pred_y)**0.5
        mape = mean_absolute_percentage_error(y, pred_y)*100
    return rmse, mape

def preprocess_dataset(seed, raw_data, noise_ratio):
    label_dim = 4
    set_seed(seed)

    shuffle_idx = np.random.permutation(raw_data.shape[0])
    data = raw_data[shuffle_idx]

    train_data = data[:2000, :]
    valid_data = data[2000:2500, :]
    test_data = data[2500:3500, :]    

    ### Noise injection
    noise_idx = np.random.choice(train_data.shape[0], int(len(train_data)*noise_ratio), replace=False)
    for n_idx in noise_idx:
        while True:
            prev_label = train_data[n_idx, -label_dim:]
            new_label = prev_label + np.random.randn(label_dim) * 60
#             if np.all(new_label <= 160) and np.all(new_label >= 30):
            if sum(new_label <= np.array([160, 160, 160, 160])) + \
                           sum(new_label >= np.array([30, 30, 30, 30])) == 8:
                train_data[n_idx, -label_dim:] = new_label
                break
        
    train_max = np.max(train_data[:, :-label_dim], axis = 0)
    train_min = np.min(train_data[:, :-label_dim], axis = 0)
    
    # Normalize all sets using the min-max from training data
    train_data[:, :-label_dim] = (train_data[:, :-label_dim] - train_min)/(train_max - train_min)
    valid_data[:, :-label_dim] = (valid_data[:, :-label_dim] - train_min)/(train_max - train_min)
    test_data[:, :-label_dim] = (test_data[:, :-label_dim] - train_min)/(train_max - train_min)
        
    Y_max = 1
    Y_min = 0
    return train_data, valid_data, test_data, Y_max, Y_min, noise_idx

def make_sp(y_tr, bw):
    sp = []
    for i in range(y_tr.shape[0]):
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(y_tr[i].reshape(-1, y_tr.shape[-1]))
        each_rate = np.exp(kde.score_samples(y_tr))
        each_rate /= np.sum(each_rate)
        sp.append(each_rate)
    return np.array(sp)

def make_sp_denorm(y_tr, bw):
    sp = []
    for i in range(y_tr.shape[0]):
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(y_tr[i].reshape(-1, y_tr.shape[-1]))
        each_rate = np.exp(kde.score_samples(y_tr))
        sp.append(each_rate)
    return np.array(sp)