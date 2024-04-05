import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.CNN import CNN
from src.models.KeylessAttention import KeylessAttention


class SK_WE(nn.Module):
    def __init__(self, feature_embed_size, lstm_hidden_size, cnn_out_channel,
                 num_joints=1, num_attribs=1, kernel_size=3,
                 batch_first=True, window_size=20, window_stride=5, n_head=4,
                 dropout=0.1, activation="relu", encoder_num_layers=2,
                 lstm_bidirectional=False, lstm_dropout=0.1,
                 attention_type='multi_head', is_attention=True):

        super(SK_WE, self).__init__()

        self.feature_embed_size = feature_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.kernel_size = kernel_size
        self.cnn_out_channel = cnn_out_channel

        self.num_attribs = num_attribs
        self.num_joints = num_joints

        self.batch_first = batch_first
        self.n_head = n_head
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.activation = activation
        self.window_size = window_size
        self.window_stride = window_stride
        self.encoder_num_layers = encoder_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.attention_type = attention_type
        self.is_attention = is_attention

        self.feature_extractor = CNN(self.num_joints, self.num_attribs, self.window_size,
                                     self.cnn_out_channel,
                                     self.feature_embed_size,
                                     self.kernel_size,
                                     dropout=self.dropout)

        self.fe_relu = nn.ReLU()
        self.fe_dropout = nn.Dropout(p=self.dropout)

        self.lstm = nn.LSTM(input_size=self.feature_embed_size,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=self.batch_first,
                            num_layers=self.encoder_num_layers,
                            bidirectional=self.lstm_bidirectional,
                            dropout=self.lstm_dropout)

        if self.lstm_bidirectional:
            if self.attention_type=='keyless':
                self.self_attn = KeylessAttention(2 * self.feature_embed_size)
            elif self.attention_type == 'multi_head':
                self.self_attn = nn.MultiheadAttention(embed_dim=2 * self.feature_embed_size,
                                                        num_heads=n_head,
                                                        dropout=self.dropout)
        else:
            if self.attention_type == 'keyless':
                self.self_attn = KeylessAttention(self.feature_embed_size)
            elif self.attention_type == 'multi_head':
                self.self_attn = nn.MultiheadAttention(embed_dim=self.feature_embed_size,
                                                        num_heads=n_head,
                                                        dropout=self.dropout)

        self.self_attn_weight = None
        self.module_fe_dropout = nn.Dropout(p=0.1)

    def set_parameter_requires_grad(self, model, fine_tune):
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, input, 
                input_mask, input_len, 
                guided_context=None):

        # print('########### Start SK MM_Module ###########')
        # print('input shape', input.size())

        x = input.view(-1, input.size(-3), input.size(-2), input.size(-1)).contiguous()
        x = x.transpose(1, 2).contiguous()
        embed = self.feature_extractor(x)

        if (self.batch_first):
            embed = embed.contiguous().view(input.size(0), -1, embed.size(-1))
        else:
            embed = embed.view(-1, input.size(1), embed.size(-1))
        #print('cnn embed shape(before matn):', embed.size())

        self.lstm.flatten_parameters()

        # input_len, idx_sort = torch.sort(input_len, descending=True)
        # idx_unsort = torch.argsort(idx_sort)
        # embed = embed.index_select(0, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        # input_packed = nn.utils.rnn.pack_padded_sequence(embed, input_len, batch_first=True)
        # r_output, (h_n, h_c) = self.lstm(input_packed)
        # r_output = nn.utils.rnn.pad_packed_sequence(r_output, batch_first=True)[0]
        #
        # r_output = r_output.index_select(0, Variable(idx_unsort))

        # transpose batch and sequence (B x S x ..) --> (S x B x ..)

        r_output, (h_n, h_c) = self.lstm(embed)

        if self.attention_type == 'multi_head':
            input_mask = input_mask[:, :r_output.size(1)]
            r_output = r_output.transpose(0, 1).contiguous()

            query = r_output
            if guided_context is not None:
                guided_context = guided_context.transpose(0, 1).contiguous()
                query = guided_context

            attn_output, self.self_attn_weight = self.self_attn(query, 
                                                        r_output, 
                                                        r_output, 
                                                        key_padding_mask=input_mask)

            attn_output = attn_output.transpose(0,1).contiguous()  # transpose batch and sequence (S x B x ..) --> (B x S x ..)
            attn_output = torch.sum(attn_output, dim=1).squeeze(dim=1)
            attn_output = F.relu(attn_output)
            attn_output = self.module_fe_dropout(attn_output)

        elif self.attention_type == 'keyless':
            attn_output, self.self_attn_weight = self.self_attn(r_output)
            attn_output = torch.sum(attn_output, 1).squeeze(1)
            attn_output = F.relu(attn_output)
            attn_output = self.module_fe_dropout(attn_output)
        else:
            attn_output = r_output[:,-1,:]

        #         print('attn_output shape', attn_output.size())
#         print('########### End SK MM_Module ###########')

        return attn_output, self.self_attn_weight
