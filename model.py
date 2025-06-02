import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.embDim = 128
        self.fc_layer = nn.Sequential(
            nn.Linear(226, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1))
        self.last_layer = nn.Linear(128, 4)
        
    def forward(self, x, last = False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = self.fc_layer(x)
        else:
            x = self.fc_layer(x)
        output = self.last_layer(x)
        if last:
            return output, x
        else:
            return output
        
    def get_embedding_dim(self):
        return self.embDim