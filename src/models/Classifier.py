import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, mm_embed_size, num_activity_types):
        super(Classifier, self).__init__()

        self.mm_embed_size = mm_embed_size
        self.num_activity_types = num_activity_types

        self.classification = nn.Sequential(nn.Linear(self.mm_embed_size, self.num_activity_types))

        self.classification.apply(self.init_weights)
    
    def forward(self, mm_embed): 
        return self.classification(mm_embed)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
