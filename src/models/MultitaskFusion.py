import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import config
from src.models.KeylessAttention import KeylessAttention
from src.models.GuidedFusion import GuidedFusion

class MultitaskFusion(nn.Module):
    def __init__(self, hparams):
        super(MultitaskFusion, self).__init__()

        self.hparams = hparams
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)
        self.multi_modal_nhead = self.hparams.multi_modal_nhead
        self.mm_embedding_attn_merge_type = self.hparams.mm_embedding_attn_merge_type
        self.modality_embedding_size = self.hparams.lstm_hidden_size

        self.task_emcoder = nn.ModuleDict()
        self.task_guided_encoder = nn.ModuleDict()
        for task_name in self.hparams.task_list:
            self.task_emcoder[task_name] = nn.Linear(1, self.hparams.indi_modality_embedding_size)
            self.task_guided_encoder[task_name] = GuidedFusion(self.hparams)

        self.mm_attn_weight = None

    def forward(self, module_out, batch):
        
        task_embeddings = {}
        for task_name in self.hparams.task_list:
            batch_size = batch[f'{task_name}_label'].size(0)
            batch[f'{task_name}_label'] = batch[f'{task_name}_label'].view(batch_size, -1).contiguous()
            
            task_embeddings[task_name] = self.task_emcoder[task_name](batch[f'{task_name}_label'])
            task_embeddings[task_name] = task_embeddings[task_name].unsqueeze(dim=1).contiguous()

        task_mm_embeddings = {}
        for task_name in self.hparams.task_list:
            task_mm_embeddings[task_name] = self.task_guided_encoder[task_name](module_out, task_embeddings[task_name])

        return task_mm_embeddings

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)