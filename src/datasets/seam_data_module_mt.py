import pytorch_lightning as pl
from src.datasets.seam_dataset_mt import *
from src.utils.log import *
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import math
from pytorch_lightning.trainer.supporters import CombinedLoader

class SEAMDataModuleMT(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.Dataset = SEAMDatasetMT
        self.collate_fn = SEAMCollator(self.hparams.modalities, self.hparams.task_list)
        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)


    # def prepare_data(self):
    def setup(self, stage=None):
        
        self.datasets = {}
        for task in self.hparams.task_list:
            for split in ['train', 'test', 'valid']:
                self.datasets[f'{task}_{split}'] = self.Dataset(hparams=self.hparams,
                                                            dataset_type=split,
                                                            task_name=task)

                self.txt_logger.log(f'{task}_{split} dataset len: {len(self.datasets[f"{task}_{split}"])}\n')

    def train_dataloader(self):
        loaders = {}
        for task in self.hparams.task_list:
            loader = DataLoader(self.datasets[f'{task}_train'],
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
            loaders[task] = loader
        return loaders

    def val_dataloader(self):
        if self.hparams.no_validation:
            return None

        loaders = {}
        for task in self.hparams.task_list:
            loader = DataLoader(self.datasets[f'{task}_valid'],
                            batch_size=min(self.hparams.batch_size, 2),
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
            loaders[task] = loader

        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def test_dataloader(self):
        if self.hparams.no_validation:
            return None

        loaders = {}
        for task in self.hparams.task_list:
            loader = DataLoader(self.datasets[f'{task}_test'],
                            batch_size=min(self.hparams.batch_size, 2),
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
            loaders[task] = loader
        
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders