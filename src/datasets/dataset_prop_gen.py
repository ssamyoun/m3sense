from torchvision.transforms import transforms

from src.config import config
from collections import defaultdict
import json


class DatasetPropGen:
    def __init__(self,
                 dataset_name):
        self.dataset_name = dataset_name

    def generate_dataset_prop(self, hparams):
        return self.get_dataset_prop(hparams)
    
    def get_dataset_prop(self, hparams):

        modality_prop = defaultdict(dict)

        for modality in hparams.modalities:
            modality_prop[modality]['cnn_out_channel'] = hparams.cnn_out_channel
            modality_prop[modality]['kernel_size'] = hparams.kernel_size
            if modality!='hnd':
                modality_prop[modality]['kernel_size'] = (1, hparams.kernel_size)

            modality_prop[modality]['feature_embed_size'] = hparams.feature_embed_size
            modality_prop[modality]['lstm_hidden_size'] = hparams.lstm_hidden_size
            modality_prop[modality]['lstm_encoder_num_layers'] = hparams.encoder_num_layers
            modality_prop[modality]['lstm_bidirectional'] = hparams.lstm_bidirectional
            modality_prop[modality]['module_embedding_nhead'] = hparams.module_embedding_nhead
            modality_prop[modality]['dropout'] = hparams.lower_layer_dropout
            modality_prop[modality]['activation'] = 'relu'
            modality_prop[modality]['fine_tune'] = hparams.fine_tune
            modality_prop[modality]['feature_pooling_kernel'] = None
            modality_prop[modality]['feature_pooling_stride'] = None
            modality_prop[modality]['feature_pooling_type'] = 'max'
            modality_prop[modality]['lstm_dropout'] = hparams.lstm_dropout

            modality_prop[modality]['window_size'] = hparams.window_size
            modality_prop[modality]['window_stride'] = hparams.window_stride
            modality_prop[modality]['sk_window_embed'] = hparams.sk_window_embed
            
            modality_prop[modality]['num_joints'] = 1
            modality_prop[modality]['num_attribs'] = 1

            modality_prop[modality]['seq_max_len'] = hparams.seq_max_len
            modality_prop[modality]['skip_frame_len'] = hparams.skip_frame_len
            modality_prop[modality]['is_rand_starting'] = False
            
        return modality_prop
