import torch.nn as nn

dropout = nn.Dropout(p=0.5)
batch_norm = nn.BatchNorm1d(num_features=128)