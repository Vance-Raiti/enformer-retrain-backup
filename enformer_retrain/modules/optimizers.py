import torch
import torch.optim as optim

Optimizers = {
    'name': 'Optimizers',
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
}
