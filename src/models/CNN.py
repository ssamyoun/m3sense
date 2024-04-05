import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self, num_features,
                 feature_len,
                 window_size,
                 cnn_out_channel,
                 feature_embed_size,
                 kernel_size,
                 dropout=0.1):
        super(CNN, self).__init__()

        self.num_features = num_features
        self.feature_len = feature_len
        self.window_size = window_size
        self.feature_embed_size = feature_embed_size
        self.cnn_out_channel = cnn_out_channel
        self.kernel_size = kernel_size
        self.max_pool_kernel_size = 2
        self.dropout = dropout

        self.conv1 = nn.Conv2d(self.feature_len, self.cnn_out_channel//2,
                               kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(self.cnn_out_channel//2)

        self.conv2 = nn.Conv2d(self.cnn_out_channel//2, self.cnn_out_channel,
                               kernel_size=self.kernel_size)
        self.conv2_bn = nn.BatchNorm2d(self.cnn_out_channel)

        conv1_out_shape = self.get_conv_output_shape((self.num_features, self.window_size), kernel_size=1)
        conv2_out_shape = self.get_conv_output_shape(conv1_out_shape, self.kernel_size)
        # print(conv1_out_shape)
        # print(conv2_out_shape)
        # print(self.kernel_size)
        # print('ll size', self.cnn_out_channel * conv2_out_shape[0] * conv2_out_shape[1])
        self.fc = nn.Linear(self.cnn_out_channel * conv2_out_shape[0] * conv2_out_shape[1],
                             self.feature_embed_size)
        self.fc_bn = nn.BatchNorm1d(self.feature_embed_size)
        self.fc_dropout = nn.Dropout(self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.conv1.bias, 0.)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, input):
        # print('########### Start CNN Module ###########')
        # print('feature_len', self.feature_len, 'num_features', self.num_features)
        # print('cnn input shape: ', input.size())

        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))

        # print('cnn output shape: ', x.size())
        x = x.view(input.size(0), -1)
        # print('cnn: x(cnn) reshape:',x.size())
        
        x = self.fc_dropout(F.relu(self.fc(x)))
        
        # print('cnn: x(fc) shape:',x.size())
        # print('########### End CNN Module ###########')
        return x

    def get_conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w
