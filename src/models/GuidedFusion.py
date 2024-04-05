import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import config
from src.models.KeylessAttention import KeylessAttention

class GuidedFusion(nn.Module):
    def __init__(self, hparams):
        super(GuidedFusion, self).__init__()

        self.hparams = hparams
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)
        self.multi_modal_nhead = self.hparams.multi_modal_nhead
        self.mm_embedding_attn_merge_type = self.hparams.mm_embedding_attn_merge_type
        self.modality_embedding_size = self.hparams.lstm_hidden_size
        self.modality_prop = self.hparams.modality_prop

        self.dropout = self.hparams.upper_layer_dropout

        self.lstm_bidirectional = False
        for modality in self.modalities:
            if (self.modality_prop[modality]['lstm_bidirectional']):
                self.lstm_bidirectional = True

        if (self.lstm_bidirectional):
            self.num_lstm_dir = 2
        else:
            self.num_lstm_dir = 1

        self.modality_embedding_size = self.num_lstm_dir * self.modality_embedding_size
        self.mm_embeddings_bn = nn.BatchNorm1d(self.num_modality)
        self.mm_embeddings_relu = nn.ReLU()
        self.mm_embeddings_dropout = nn.Dropout(p=self.dropout)

        if self.num_modality>1 and self.hparams.mm_fusion_attention_type is not None:
            self.mm_mhattn = nn.MultiheadAttention(embed_dim=self.modality_embedding_size,
                                                num_heads=self.multi_modal_nhead,
                                                dropout=self.dropout)

            self.mm_mhattn_bn = nn.BatchNorm1d(self.num_modality)
            self.mm_mhattn_relu = nn.ReLU()
            self.mm_mhattn_dropout = nn.Dropout(p=self.dropout)

        
        if (self.mm_embedding_attn_merge_type == 'sum'):
            mm_input_dim = self.num_lstm_dir * self.modality_embedding_size
            mm_output_dim = self.hparams.indi_modality_embedding_size
            self.fc_output1 = nn.Linear(mm_input_dim, mm_output_dim)
            # self.fc_output2 = nn.Linear(mm_input_dim, mm_output_dim)
            
        else:
            mm_input_dim = self.num_modality * self.modality_embedding_size
            mm_output_dim = self.hparams.indi_modality_embedding_size
            self.fc_output1 = nn.Linear(mm_input_dim, mm_output_dim)
        
        self.mm_attn_weight = None

    def forward(self, module_out, guided_context):
        mm_embeddings = []
        for modality in self.modalities:
            mm_embeddings.append(module_out[modality])

        mm_embeddings = torch.stack(mm_embeddings, dim=1).contiguous()
        mm_embeddings = F.relu(self.mm_embeddings_bn(mm_embeddings))
        nbatches = mm_embeddings.shape[0]
        
        if self.num_modality>1 and (self.hparams.mm_fusion_attention_type=='multi_head' or self.hparams.mm_fusion_attention_type=='keyless'):
            if self.hparams.mm_fusion_attention_type=='multi_head':
                # transpose batch and sequence (B x S x ..) --> (S x B x ..)
                mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()

                query = mm_embeddings
                if guided_context is not None:
                    guided_context = guided_context.transpose(0, 1).contiguous()
                    query = guided_context

                # remove the modality mask batch['modality_mask']
                # so that the model can learn by itself that 
                # how it can learn in the presence of missing modality
                mm_embeddings, self.mm_attn_weight = self.mm_mhattn(query, 
                                                                mm_embeddings, 
                                                                mm_embeddings)
                # transpose batch and sequence (S x B x ..) --> (B x S x ..)
                mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()  
                if guided_context is None:
                    mm_embeddings = self.mm_mhattn_bn(mm_embeddings)

            elif self.hparams.mm_fusion_attention_type=='keyless':
                mm_embeddings, self.mm_attn_weight = self.mm_mhattn(mm_embeddings)

            mm_embeddings = self.mm_mhattn_dropout(self.mm_mhattn_relu(mm_embeddings))
        
        if self.mm_embedding_attn_merge_type == 'sum':
            mm_embeddings = torch.sum(mm_embeddings, dim=1).squeeze(dim=1)

        mm_embeddings = mm_embeddings.contiguous().view(nbatches, -1)
        mm_embed = F.relu(self.fc_output1(mm_embeddings))

        return mm_embed

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)