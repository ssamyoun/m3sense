import pytorch_lightning as pl
from src.datasets.seam_dataset import *
from src.utils.log import *
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import math

class SEAMDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.Dataset = SEAMDataset
        self.collate_fn = SEAMCollator(self.hparams.modalities)

        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)


    # def prepare_data(self):
    def setup(self, stage=None):
        
        if (self.hparams.data_split_type=='fixed_session'):
            self.train_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='train')

            self.valid_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='valid')

            self.test_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='test')
        
        elif self.hparams.data_split_type=='session':
            full_dataset = self.Dataset(hparams=self.hparams)
            dataset_len = len(full_dataset)
            self.hparams.dataset_len = dataset_len

            test_len = math.floor(dataset_len*self.hparams.test_split_pct)
            valid_len = math.floor((dataset_len-test_len)*self.hparams.valid_split_pct)
            train_len = dataset_len - valid_len - test_len

            self.train_dataset, self.valid_dataset, self.test_dataset = random_split(full_dataset,
                                                                            [train_len, valid_len, test_len])

        # self.txt_logger.log(f'train subject ids: {sorted(self.train_dataset.data[self.hparams.train_element_tag].unique())}\n')
        # self.txt_logger.log(f'valid subject ids: {sorted(self.valid_dataset.data[self.hparams.train_element_tag].unique())}\n')
        # self.txt_logger.log(f'test subject ids: {sorted(self.test_dataset.data[self.hparams.train_element_tag].unique())}\n')
        self.txt_logger.log(f'**train dataset len: {len(self.train_dataset)}\n')
        self.txt_logger.log(f'**valid dataset len: {len(self.valid_dataset)}\n')
        self.txt_logger.log(f'**test dataset len: {len(self.test_dataset)}\n')

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
        return loader

    def val_dataloader(self):
        if self.hparams.no_validation:
            return None
            
        loader = DataLoader(self.test_dataset,
                            batch_size=min(self.hparams.batch_size, 2),
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=min(self.hparams.batch_size, 2),
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
        return loader