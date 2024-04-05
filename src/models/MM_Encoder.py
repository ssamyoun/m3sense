import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.SK_WE import SK_WE
from src.config import config
from src.models.KeylessAttention import KeylessAttention

class MM_Encoder(nn.Module):
    def __init__(self, hparams):

        super(MM_Encoder, self).__init__()
        self.hparams = hparams
        self.modality_prop = self.hparams.modality_prop
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)
        self.num_labels = self.hparams.num_labels

        self.multi_modal_nhead = self.hparams.multi_modal_nhead
        self.mm_embedding_attn_merge_type = self.hparams.mm_embedding_attn_merge_type
        self.lstm_bidirectional = self.hparams.lstm_bidirectional
        self.modality_embedding_size = self.hparams.lstm_hidden_size

        self.dropout = self.hparams.upper_layer_dropout

        self.nn_init_type = 'xu'
        self.batch_first = True
        self.activation = 'relu'

        self.mm_module = nn.ModuleDict()
        for modality in self.modalities:
            self.mm_module[modality] = SK_WE(num_joints=self.modality_prop[modality]['num_joints'],
                                                num_attribs=self.modality_prop[modality]['num_attribs'],
                                                cnn_out_channel=self.modality_prop[modality][
                                                    'cnn_out_channel'],
                                                feature_embed_size=self.modality_prop[modality][
                                                    'feature_embed_size'],
                                                kernel_size=self.modality_prop[modality]['kernel_size'],
                                                lstm_hidden_size=self.modality_prop[modality][
                                                    'lstm_hidden_size'],
                                                batch_first=self.batch_first,
                                                window_size=self.modality_prop[modality]['window_size'],
                                                window_stride=self.modality_prop[modality]['window_stride'],
                                                n_head=self.modality_prop[modality][
                                                    'module_embedding_nhead'],
                                                dropout=self.modality_prop[modality]['dropout'],
                                                activation=self.modality_prop[modality]['activation'],
                                                encoder_num_layers=self.modality_prop[modality][
                                                    'lstm_encoder_num_layers'],
                                                lstm_bidirectional=self.modality_prop[modality][
                                                    'lstm_bidirectional'],
                                                lstm_dropout=self.modality_prop[modality]['lstm_dropout'],
                                                attention_type=self.hparams.unimodal_attention_type)


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
            if self.hparams.mm_fusion_attention_type=='multi_head':
                self.mm_mhattn = nn.MultiheadAttention(embed_dim=self.modality_embedding_size,
                                                    num_heads=self.multi_modal_nhead,
                                                    dropout=self.dropout)
            elif self.hparams.mm_fusion_attention_type=='keyless':
                self.mm_mhattn = KeylessAttention(self.modality_embedding_size)

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
            # self.fc_output2 = nn.Linear(mm_input_dim, mm_output_dim)
        
        self.module_out = {}
        self.module_attn_weights = {}
        self.mm_attn_weight = None
        self.reset_parameters()

    def forward(self, batch, guided_context=None):
        
        for modality in self.modalities:
            # print(f'{modality}: {batch[modality].shape}')
            tm_attn_output, self.module_attn_weights[modality] = self.mm_module[modality](batch[modality],
                                                                            batch[modality + config.modality_mask_suffix_tag],
                                                                            batch[modality + config.modality_seq_len_tag],
                                                                            guided_context)
            self.module_out[modality] = tm_attn_output

        mm_embeddings = []
        for modality in self.modalities:
            mm_embeddings.append(self.module_out[modality])

        mm_embeddings = torch.stack(mm_embeddings, dim=1).contiguous()
        mm_embeddings = F.relu(self.mm_embeddings_bn(mm_embeddings))
        nbatches = mm_embeddings.shape[0]
        
        if self.num_modality>1 and (self.hparams.mm_fusion_attention_type=='multi_head' or self.hparams.mm_fusion_attention_type=='keyless' ):
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
        # mm_embed = F.relu(self.fc_output2(mm_embed))

        return self.module_out, mm_embed

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)

        # nn.init.xavier_uniform_(self.fc_output2.weight)
        # nn.init.constant_(self.fc_output2.bias, 0.)