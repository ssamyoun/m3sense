import torch
import torch.nn as nn 
import torch.nn.functional as F

class GaussianNoise:
    def __init__(self, noise_level, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        self.noise_level = noise_level
        
    def __call__(self, x):
        noise = torch.randn(x.size()) * self.std + self.mean
        x = (1-self.noise_level) * x + self.noise_level * noise
        return x 

class RandomNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level
        
    def __call__(self, x):
        with torch.no_grad():
            drop = nn.Dropout(self.noise_level)
            noise = drop(torch.ones(x.shape))
            x = x * noise
        return x 