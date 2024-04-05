import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from src.config import config
from src.utils.log import *
from src.models.MM_Encoder import MM_Encoder
from .Classifier import Classifier
from src.models.Teacher_Trainer import *
from src.utils.log import TextLogger
from src.utils.model_checkpointing import ModelCheckpointing
from src.utils.training_utils import *
from collections import Counter
from src.utils.model_saving import *

from src.models.kd_logits import Logits 


class StudentTrainerAlignment(pl.LightningModule):
    def __init__(self,
                 hparams):

        super(StudentTrainerAlignment, self).__init__()

        self.hparams.update(vars(hparams))
        self.teacher_model = TeacherTrainer(hparams=hparams)
        teacher_resume_path = f'{hparams.model_save_base_dir}/{hparams.resume_teacher_checkpoint_filename}'
        self.teacher_model, _ = load_model(self.teacher_model, teacher_resume_path, strict_load=False)


        self.modality_prop = self.hparams.modality_prop
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)
        self.mm_embedding_attn_merge_type = self.hparams.mm_embedding_attn_merge_type
        self.lstm_bidirectional = self.hparams.lstm_bidirectional
        self.modality_embedding_size = self.hparams.lstm_hidden_size
        self.num_labels = self.hparams.num_labels
        self.weights_loss = []

        # build learning model
        self.mm_encoder = MM_Encoder(self.hparams)

        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_logits = Logits()

        self.config_checkpointer()

        self.is_multi_optimizers = False
        self.test_log = None
        self.mm_embed = None
        self.module_out = None
    
    def forward(self, batch):
        self.module_out, self.mm_embed = self.mm_encoder(batch)
        return self.module_out, self.mm_embed

    def set_parameter_requires_grad(self, model, is_require):
        for param in model.parameters():
            param.requires_grad = is_require

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.eval_step(batch, batch_idx, pre_log_tag='train', optimizer_idx=optimizer_idx)

    def training_epoch_end(self, outputs):
        results = cal_metrics(outputs, self.pl_metrics_list, self.pl_metrics, 
                            stage_tag='train', trainer=self.trainer, 
                            device=self.device,
                            is_multi_optimizers=self.is_multi_optimizers)
        self.log_metrics(results)
        model = self.get_model()
        self.train_model_checkpointing.update_metric_save_ckpt(results, model, self.current_epoch, self.trainer)
        
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, pre_log_tag='valid')

    def validation_epoch_end(self, outputs):
        results = cal_metrics(outputs, self.pl_metrics_list, 
                            self.pl_metrics, stage_tag='valid',
                            trainer=self.trainer, device=self.device,
                            is_multi_optimizers=self.is_multi_optimizers)
        self.log_metrics(results)
        model = self.get_model()
        self.valid_model_checkpointing.update_metric_save_ckpt(results, model, self.current_epoch, self.trainer)
        

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, pre_log_tag='test')

    def test_epoch_end(self, outputs):
        results = cal_metrics(outputs, self.pl_metrics_list, 
                            self.pl_metrics, stage_tag='test',
                            trainer=self.trainer, device=self.device,
                            is_multi_optimizers=self.is_multi_optimizers)
        self.log_metrics(results)
        self.test_log = results
        self.txt_logger.log(f'{str(results)}\n')
        
    def eval_step(self, batch, batch_idx, pre_log_tag, optimizer_idx=0):
        
        labels = batch['label']
        batch_size = batch['label'].size(0)

        _ , student_mm_embed = self(batch)
        
        with torch.no_grad():
            teacher_output = self.teacher_model(batch)
            teacher_mm_embed = self.teacher_model.mm_embed.detach()

        loss = self.kd_logits(student_mm_embed, teacher_mm_embed)
        self.log(f'{pre_log_tag}_encoder_loss', loss)

        # _, preds = torch.max(F.softmax(har_output, dim=1), 1)
        # metric_results = {}
        # for metric in self.pl_metrics_list:
        #     metric_key = f'{pre_log_tag}_{metric}'
        #     metric_results[metric_key] = self.pl_metrics[metric_key](preds, labels)

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer_mm_encoder = torch.optim.AdamW(self.mm_encoder.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler_mm_encoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_mm_encoder,
                                                                        T_0=self.hparams.cycle_length,
                                                                        T_mult=self.hparams.cycle_mul)                                                                      
        return [optimizer_mm_encoder], [lr_scheduler_mm_encoder]
    
    def config_checkpointer(self):
        # define the metrics and the checkpointing mode
        self.metrics_mode_dict = {'loss': 'min',
                            'accuracy': 'max',
                            'f1_scores': 'max',
                            'precision': 'max',
                            'recall_scores': 'max'}
        train_metrics_save_ckpt_mode = {'epoch_train_loss': True}
        valid_metrics_save_ckpt_mode = {'epoch_valid_loss': True,
                                    'epoch_valid_accuracy': True}
        train_metrics_mode_dict = {}
        valid_metrics_mode_dict = {}
        train_metrics = []
        valid_metrics = []

        self.pl_metrics_list = ['accuracy']
        self.pl_metrics = nn.ModuleDict()

        for metric in self.metrics_mode_dict:
            train_metrics.append(f'epoch_train_{metric}')
            valid_metrics.append(f'epoch_valid_{metric}')
            train_metrics_mode_dict[f'epoch_train_{metric}'] = self.metrics_mode_dict[metric]
            valid_metrics_mode_dict[f'epoch_valid_{metric}'] = self.metrics_mode_dict[metric]
            
        stages = ['train', 'valid', 'test']
        for metric in self.pl_metrics_list:
            for stage in stages:
                self.pl_metrics[f'{stage}_{metric}'] = get_pl_metrics(metric, self.hparams.num_labels)

        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)
        self.train_model_checkpointing = ModelCheckpointing(self.hparams.model_save_base_dir,
                                                self.hparams.model_checkpoint_filename,
                                                train_metrics,
                                                train_metrics_save_ckpt_mode,
                                                train_metrics_mode_dict,
                                                self.txt_logger)
        
        self.valid_model_checkpointing = ModelCheckpointing(self.hparams.model_save_base_dir,
                                                self.hparams.model_checkpoint_filename,
                                                valid_metrics,
                                                valid_metrics_save_ckpt_mode,
                                                valid_metrics_mode_dict,
                                                self.txt_logger)
    
    
    def get_model(self):
        is_dp_module = isinstance(self, (LightningDistributedDataParallel,
                                         LightningDataParallel))
        model = self.module if is_dp_module else self
        return model

    def log_metrics(self, results):
        for metric in results:
            self.log(metric, results[metric])