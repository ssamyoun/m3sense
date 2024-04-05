import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.SK_WE import SK_WE
from src.config import config
from src.models.KeylessAttention import KeylessAttention

class MM_Encoder_MT(nn.Module):
    def __init__(self, hparams):

        super(MM_Encoder_MT, self).__init__()
        self.hparams = hparams
        self.modality_prop = self.hparams.modality_prop
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)

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
        
        self.module_out = {}
        self.module_attn_weights = {}
        
        # self.reset_parameters()

    def forward(self, batch, guided_context=None):
        
        for modality in self.modalities:
            # print(f'{modality}: {batch[modality].shape}')
            tm_attn_output, self.module_attn_weights[modality] = self.mm_module[modality](batch[modality],
                                                                            batch[modality + config.modality_mask_suffix_tag],
                                                                            batch[modality + config.modality_seq_len_tag],
                                                                            guided_context)
            self.module_out[modality] = tm_attn_output

        return self.module_out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)

        # nn.init.xavier_uniform_(self.fc_output2.weight)
        # nn.init.constant_(self.fc_output2.bias, 0.)