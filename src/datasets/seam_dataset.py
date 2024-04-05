import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.config import config
from src.utils.noises import *
from src.utils.log import *


class SEAMDataset(Dataset):

    def __init__(self,
                 hparams,
                 dataset_type='train',
                 restricted_ids=None, restricted_labels=None,
                 allowed_ids=None, allowed_labels=None,
                 noisy_sampler=None):
        self.hparams = hparams
        self.task_name = hparams.task_name
        self.dataset_type = dataset_type
        self.base_dir = self.hparams.data_file_dir_base_path
        self.modality_prop = self.hparams.modality_prop
        self.restricted_ids = restricted_ids
        self.restricted_labels = restricted_labels
        self.allowed_ids = allowed_ids
        self.allowed_labels = allowed_labels
        self.modalities = self.hparams.modalities
        self.noisy_sampler = noisy_sampler

        self.load_data()

    def load_data(self):

        self.data = {}
        for modality in self.modalities:
            if(self.hparams.dataset_subset_name is None):
                self.data[modality] = pd.read_csv(f'{self.base_dir}/{modality}.csv')
            else:
                self.data[modality] = pd.read_csv(f'{self.base_dir}/{self.dataset_type}_{self.hparams.dataset_subset_name}_{modality}.csv')

            if(not self.hparams.is_student_training):
                if(self.hparams.student_target_task_name is not None):
                    self.data[modality] = self.data[modality][self.data[modality]['application']==self.hparams.student_target_task_name]
                else:
                    self.data[modality] = self.data[modality][self.data[modality]['application']==self.task_name]
            self.num_labels = len(self.data[modality].label.unique())

            if self.restricted_ids is not None:
                self.data[modality] = self.data[modality][~self.data[modality][self.restricted_labels].isin(self.restricted_ids)]

            if self.allowed_ids is not None:
                self.data[modality] = self.data[modality][self.data[modality][self.allowed_labels].isin(self.allowed_ids)]

            self.data[modality].reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data[self.modalities[0]])

    def __getitem__(self, idx):
        data = {}
        skip_frame_len_dict = {}
        for modality in self.modalities:
            skip_frame_len_dict[modality] = int(self.modality_prop[modality]['skip_frame_len'])
        data = self.prepare_data_element(idx, skip_frame_len_dict)
        return data

    def prepare_data_element(self, idx, skip_frame_len_dict=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        activity_types = []
        is_noisy = False 
        noisy_modalities = None
        noisy_modality_idx = len(self.modalities)
        if self.noisy_sampler is not None:
            noisy_sample = self.noisy_sampler.sample()
            if noisy_sample[0]==1 and (self.hparams.noisy_modalities is None):
                is_noisy = True
                noisy_modality_idx = random.randint(0, len(self.modalities)-1)
                noisy_modalities = [self.modalities[noisy_modality_idx]]
            elif self.hparams.noisy_modalities is not None:
                is_noisy = True
                noisy_modalities = self.hparams.noisy_modalities

        addNoise = {}
        if noisy_modalities is not None:
            for modality in self.modalities:
                if (modality in noisy_modalities) and is_noisy and (self.hparams.noise_type is not None):
                    if(self.hparams.noise_type=='random'):
                        addNoise[modality] = RandomNoise(noise_level=self.hparams.noise_level[modality])
                    elif(self.hparams.noise_type=='gaussian'):
                        addNoise[modality] = GaussianNoise(noise_level=self.hparams.noise_level[modality])

        data = {}
        modality_mask = []
        for modality in self.modalities:
            seq, seq_len = self.get_data(modality, idx)

            if is_noisy and (self.hparams.noise_type is not None):
                if modality in noisy_modalities:
                    seq = addNoise[modality](seq)

            data[modality] = seq
            data[modality + config.modality_seq_len_tag] = seq_len

            if (seq_len == 0):
                modality_mask.append(True)
            else:
                modality_mask.append(False)

        modality_mask = torch.from_numpy(np.array(modality_mask)).bool()

        label = self.data[self.modalities[0]].loc[idx, ['label']][0]

        if(int(label)==2 and (self.hparams.student_target_task_name=='Anxiety' or self.hparams.task_name=='Anxiety')):
            data['label'] = 1    
        else:
            data['label'] = int(label)

        domain_name = self.data[self.modalities[0]].loc[idx, ['application']][0]
        data['domain_label'] = config.domain_labels[domain_name]
        data['modality_mask'] = modality_mask
        data['is_noisy'] = 1 if is_noisy else 0
        data['noisy_modality'] = noisy_modality_idx

        return data

    def get_data(self, modality, idx):
        data_filepath = f'{self.base_dir}/{self.data[modality].loc[idx, "file_location"]}'
        startIndex = self.data[modality].loc[idx, 'startRowIndex']
        endIndex = self.data[modality].loc[idx, 'endRowIndex']
        
        
        temp_df = pd.read_csv(data_filepath)
        if('Anxiety' in data_filepath):
            data_column_key = modality
        else:
            data_column_key = self.data[modality].loc[idx, 'file_key']
        data_item = temp_df[data_column_key][startIndex:endIndex+1].to_numpy()
        
        seq = torch.tensor(data_item, dtype=torch.float)
        seq = seq.unsqueeze(dim=1).contiguous()
        # print('#### pre',modality,seq.shape, startIndex, endIndex, data_filepath)
        if self.modality_prop[modality]['sk_window_embed'] and seq.shape[0]!=0:
            # print(f'seq shape {seq.shape}')
            if(seq.shape[0] < self.modality_prop[modality]['window_size']):
                seq = seq.repeat(self.modality_prop[modality]['window_size']-seq.shape[0]+1, 1)
            seq = self.split_seq(seq,
                                self.modality_prop[modality]['window_size'],
                                self.modality_prop[modality]['window_stride'])
        else:
            seq = torch.zeros(
                        (self.modality_prop[modality]['seq_max_len'], 1, 5)).float()
            print('#### pre',modality, self.data[modality].loc[idx, 'id'],seq.shape, startIndex, endIndex, data_filepath)
        # print('#### post',modality,seq.shape)
        seq = seq.unsqueeze(dim=-2).contiguous()
        seq_len = seq.size(0)
        if (self.modality_prop[modality]['seq_max_len'] is not None) and (seq_len > self.modality_prop[modality]['seq_max_len']):
            seq = seq[:self.modality_prop[modality]['seq_max_len'], :]

        return seq, seq.size(0)

    def split_seq(self, seq, window_size, window_stride):
        return seq.unfold(dimension=0, size=window_size, step=window_stride)

def gen_mask(seq_len, max_len):
   return torch.arange(max_len) > seq_len

class SEAMCollator:

    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, batch):
        batch_size = len(batch)
        data = {}
        for modality in self.modalities:
            data[modality] = pad_sequence([batch[bin][modality] for bin in range(batch_size)], batch_first=True)
            data[modality + config.modality_seq_len_tag] = torch.tensor(
                    [batch[bin][modality + config.modality_seq_len_tag] for bin in range(batch_size)],
                    dtype=torch.float)
            
            seq_max_len = data[modality + config.modality_seq_len_tag].max()
            seq_mask = torch.stack(
            [gen_mask(seq_len, seq_max_len) for seq_len in data[modality + config.modality_seq_len_tag]],
                dim=0)
            data[modality + config.modality_mask_tag] = seq_mask

        data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['domain_label'] = torch.tensor([batch[bin]['domain_label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['is_noisy'] = torch.tensor([batch[bin]['is_noisy'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['noisy_modality'] = torch.tensor([batch[bin]['noisy_modality'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
        return data