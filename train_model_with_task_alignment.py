
# !/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from datetime import datetime
import torch
import wandb
from pytorch_lightning import loggers, Trainer, seed_everything
from sklearn.model_selection import LeaveOneOut

from src.datasets.dataset_prop_gen import DatasetPropGen
from src.datasets.seam_dataset_mt import *
from src.datasets.seam_data_module_mt import *
from src.models.Multitask_Trainer import *
from src.utils.model_saving import *
from src.utils.debug_utils import *
from src.utils.log import TextLogger
from src.config import config

from collections import defaultdict
import numpy as np
import torch
import json

test_metrics = {}

def main(args):
    seed_everything(333)

    txt_logger = TextLogger(args.log_base_dir, 
                            args.log_filename,
                            print_console=True)

    args.feature_embed_size = args.indi_modality_embedding_size
    args.lstm_hidden_size = args.indi_modality_embedding_size

    if args.model_checkpoint_filename is None:
        args.model_checkpoint_filename = f'{args.model_checkpoint_prefix}.pth'
    
    args.temp_model_checkpoint_filename = args.model_checkpoint_filename
    args.test_models = args.test_models.strip().split(',')
    args.test_metrics = args.test_metrics.strip().split(',')
    args.task_list = args.task_list.strip().split(',')

    for test_model in args.test_models:
        test_metrics[f'{test_model}'] = defaultdict(list)
    

    txt_logger.log(f'model_checkpoint_prefix:{args.model_checkpoint_prefix}\n')
    txt_logger.log(f'model_checkpoint_filename:{args.model_checkpoint_filename}, resume_checkpoint_filename:{args.resume_checkpoint_filename}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_logger.log(f'pytorch version: {torch.__version__}\n')
    txt_logger.log(f'GPU Availability: {device}, gpus: {args.gpus}\n')

    # Set dataloader class and prop 
    args.modalities = args.modalities.strip().split(',')
    Dataset = SEAMDatasetMT
    args.training_label = 'pid'
    collate_fn = SEAMCollator(args.modalities,args.task_list)

    datasetPropGen = DatasetPropGen(args.dataset_name)
    args.modality_prop = datasetPropGen.generate_dataset_prop(args)
    
    if args.exe_mode=='dl_debug':
        train_dataset = Dataset(args)
        debug_dataloader(train_dataset, collate_fn, args, last_batch=-1)
        return

    if args.dataset_name=='seam':

        if args.resume_checkpoint_filename is not None:
            args.resume_checkpoint_filepath = f'{args.model_save_base_dir}/{args.resume_checkpoint_filename}'
            if os.path.exists(args.resume_checkpoint_filepath):
                args.resume_training = True
            else:
                args.resume_training = False

        loggers_list = []
        if (args.tb_writer_name is not None) and (args.exe_mode=='train'):
            loggers_list.append(loggers.TensorBoardLogger(save_dir=args.log_base_dir, 
                                name=f'{args.tb_writer_name}'))
        if (args.wandb_log_name is not None) and (args.exe_mode=='train'):
            loggers_list.append(loggers.WandbLogger(save_dir=args.log_base_dir, 
                            log_model="all",
                            name=f'{args.wandb_log_name}',
                            entity=f'{args.wandb_entity}',
                            project='Multimodal-Domain-Adaptation'))

        args.train_restricted_ids = None
        args.train_restricted_labels = None
        args.train_allowed_ids = None
        args.train_allowed_labels = None

        args.valid_restricted_ids = None
        args.valid_restricted_labels = None
        args.valid_allowed_ids = None
        args.valid_allowed_labels = None

        args.test_restricted_ids = None
        args.test_restricted_labels = None
        args.test_allowed_ids = None
        args.test_allowed_labels = None

        args.model_checkpoint_filename = f'{args.model_checkpoint_filename}'
        txt_logger.log(str(args), print_console=args.log_model_archi)

        start_training(args, txt_logger, loggers_list)        

def start_training(args, txt_logger, loggers=None):

    txt_logger.log(f"\n\n$$$$$$$$$ Start training $$$$$$$$$\n\n")
    dataModule = SEAMDataModuleMT(args)
    model = MultitaskTrainer(hparams=args)

    if args.log_model_archi:
        txt_logger.log(str(model))

    if args.resume_training:
        model, _ = load_model(model, args.resume_checkpoint_filepath, strict_load=False)
        txt_logger.log(f'Reload model from chekpoint: {args.resume_checkpoint_filename}\n model_checkpoint_filename: {args.model_checkpoint_filename}\n')


    if args.compute_mode=='gpu':
        trainer = Trainer.from_argparse_args(args,gpus=args.gpus, 
                    distributed_backend=args.distributed_backend,
                    max_epochs=args.epochs,
                    logger=loggers,
                    checkpoint_callback=False,
                    precision=args.float_precision,
                    limit_train_batches=args.train_percent_check,
                    num_sanity_val_steps=args.num_sanity_val_steps,
                    limit_val_batches=args.val_percent_check,
                    fast_dev_run=args.fast_dev_run)

    if args.only_testing:
        trainer.test(model, datamodule=dataModule)
    else:
        if args.lr_find:
            print('Start Learning rate finder')
            lr_trainer = Trainer()
            lr_finder = lr_trainer.lr_find(model)
            fig = lr_finder.plot(suggest=True)
            fig.show()
            new_lr = lr_finder.suggestion()
            txt_logger.log(str(new_lr))
            model.hparams.learning_rate = new_lr

        trainer.fit(model, datamodule=dataModule)
        if args.is_test:
            txt_logger.log(f"\n\n$$$$$$$$$ Start testing $$$$$$$$$\n\n")
            for test_model in args.test_models:
                trainer = None
                model = None 
                trainer = Trainer.from_argparse_args(args,gpus=args.gpus, 
                            distributed_backend=None,
                            max_epochs=args.epochs,
                            logger=None,
                            checkpoint_callback=False,
                            precision=args.float_precision)

                model = MultitaskTrainer(hparams=args)
                ckpt_filename = f'best_epoch_{test_model}_{args.model_checkpoint_filename}'
                ckpt_filepath = f'{args.model_save_base_dir}/{ckpt_filename}'
                if not os.path.exists(ckpt_filepath):
                    txt_logger.log(f'Skip testing model for chekpoint({ckpt_filepath}) is not found\n')
                    continue 
                model, _ = load_model(model, ckpt_filepath, strict_load=False)
                # model = TeacherTrainer.load_from_checkpoint(ckpt_filepath)
                model.eval()
                txt_logger.log(f'Reload testing model from chekpoint: {ckpt_filepath}\n')
                txt_logger.log(f'{test_model}')
                trainer.test(model, datamodule=dataModule)
                
                # test_log = model.test_log 
                # for test_metric in args.test_metrics:
                #     test_metrics[f'{test_model}'][f'test_{test_metric}'].append(test_log[f'test_{test_metric}'])
                
                trainer = None
                model = None
                torch.cuda.empty_cache()

    trainer = None
    model = None 
    torch.cuda.empty_cache()

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    parser.add_argument("-compute_mode", "--compute_mode", help="compute_mode",
                        default='gpu')
    parser.add_argument("--fast_dev_run", help="fast_dev_run",
                        action="store_true", default=False)    
    parser.add_argument("-num_nodes", "--num_nodes", help="num_nodes",
                        type=int, default=1)
    parser.add_argument("-distributed_backend", "--distributed_backend", help="distributed_backend",
                        default='ddp_spawn')
    parser.add_argument("--gpus", help="number of gpus or gpus list",
                        default="-1")
    parser.add_argument("--float_precision", help="float precision",
                        type=int, default=32)
    parser.add_argument("--dataset_name", help="dataset_name",
                        default=None)
    parser.add_argument("--dataset_subset_name", help="dataset_subset_name",
                        default=None)
    parser.add_argument("-ws", "--window_size", help="windows size",
                        type=int, default=5)
    parser.add_argument("-wst", "--window_stride", help="windows stride",
                        type=int, default=5)
    parser.add_argument("-ks", "--kernel_size", help="kernel size",
                        type=int, default=3)
    parser.add_argument("-bs", "--batch_size", help="batch size",
                        type=int, default=2)
    parser.add_argument("-nw", "--num_workers", help="num_workers",
                        type=int, default=2)
    parser.add_argument("-ep", "--epochs", help="epoch per validation cycle",
                        type=int, default=200)
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument("-sml", "--seq_max_len", help="maximum sequence length",
                        type=int, default=300)
    parser.add_argument("-rt", "--resume_training", help="resume training",
                        action="store_true", default=False)
    parser.add_argument("-sl", "--strict_load", help="partially or strictly load the saved model",
                        action="store_true", default=False)
    parser.add_argument("-vt", "--validation_type", help="validation_type",
                        default='person')
    parser.add_argument("-tvp", "--total_valid_persons", help="Total valid persons",
                        type=int, default=1)
    parser.add_argument("-dfp", "--data_file_dir_base_path", help="data_file_dir_base_path",
                        default=None)
    parser.add_argument("-cout", "--cnn_out_channel", help="CNN out channel size",
                        type=int, default=64)
    parser.add_argument("-fes", "--feature_embed_size", help="CNN feature embedding size",
                        type=int, default=256)
    parser.add_argument("-lhs", "--lstm_hidden_size", help="LSTM hidden embedding size",
                        type=int, default=256)
    parser.add_argument("--indi_modality_embedding_size", help="indi_modality_embedding_size",
                        type=int, default=None)
    parser.add_argument("-madf", "--matn_dim_feedforward", help="matn_dim_feedforward",
                        type=int, default=256)
    parser.add_argument("-menh", "--module_embedding_nhead", help="unimodal modal embedding multi-head attention nhead",
                        type=int, default=1)
    parser.add_argument("-mmnh", "--multi_modal_nhead", help="multi-modal embeddings multi-head attention nhead",
                        type=int, default=1)
    parser.add_argument("-enl", "--encoder_num_layers", help="LSTM encoder layer",
                        type=int, default=2)
    parser.add_argument("-lstm_bi", "--lstm_bidirectional", help="LSTM bidirectional [True/False]",
                        action="store_true", default=False)
    parser.add_argument("-skwe", "--sk_window_embed",
                        help="is skeleton and sensor data segmented by window [True/False]",
                        action="store_true", default=True)
    parser.add_argument("-fine_tune", "--fine_tune", help="Visual feature extractor fine tunning",
                        action="store_true", default=False)

    parser.add_argument("-mcp", "--model_checkpoint_prefix", help="model checkpoint filename prefix",
                        default='seam')
    parser.add_argument("-mcf", "--model_checkpoint_filename", help="model checkpoint filename",
                        default=None)
    parser.add_argument("-rcf", "--resume_checkpoint_filename", help="resume checkpoint filename",
                        default=None)

    parser.add_argument("--unimodal_attention_type",
                        help="unimodal_attention_type [multi_head/keyless]",
                        default=None)
    parser.add_argument("--mm_fusion_attention_type",
                        help="mm_fusion_attention_type [multi_head/keyless]",
                        default=None)
    parser.add_argument("--mm_fusion_attention_nhead",help="mm_fusion_attention_nhead",
                        type=int, default=1)
    parser.add_argument("--mm_fusion_attention_dropout",help="mm_fusion_attention_dropout",
                        type=float, default=0.1)
    parser.add_argument("--mm_fusion_dropout",help="mm_fusion_attention_dropout",
                        type=float, default=0.1)

    parser.add_argument("-mmattn_type", "--mm_embedding_attn_merge_type",
                        help="mm_embedding_attn_merge_type [concat/sum]",
                        default='sum')

    parser.add_argument("-logf", "--log_filename", help="execution log filename",
                        default='exe_seam.log')
    parser.add_argument("-logbd", "--log_base_dir", help="execution log base dir",
                        default='log/seam')
    parser.add_argument("-final_log", "--final_log_filename", help="Final result log filename",
                        default='final_results_seam.log')
    parser.add_argument("-tb_wn", "--tb_writer_name", help="tensorboard writer name",
                        default=None)
    parser.add_argument("-wdbln", "--wandb_log_name", help="wandb_log_name",
                        default=None)
    parser.add_argument("--wandb_entity", help="wandb_entity",
                        default='crg')
    parser.add_argument("--log_model_archi", help="log model",
                        action="store_true", default=False)

    parser.add_argument("--executed_number_it", help="total number of executed iteration",
                        type=int, default=-1)
    parser.add_argument("--end_val_it", help="end_val_it",
                        type=int, default=-1)
    parser.add_argument("-vpi", "--valid_person_index", help="valid person index",
                        type=int, default=0)
    parser.add_argument("-msbd", "--model_save_base_dir", help="model_save_base_dir",
                        default="trained_model")
    parser.add_argument("-exe_mode", "--exe_mode", help="exe_mode[dl_test/train]",
                        default='train')
    parser.add_argument("--train_percent_check", help="train_percent_check",
                        type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", help="num_sanity_val_steps",
                        type=int, default=5)
    parser.add_argument("--val_percent_check", help="val_percent_check",
                        type=float, default=1.0)
    parser.add_argument("--no_validation", help="no_validation",
                        action="store_true", default=False)
    parser.add_argument("--slurm_job_id", help="slurm_job_id",
                        default=None)

    # Data preprocessing
    parser.add_argument("--modalities", help="modalities",
                        default=None)
    parser.add_argument("--data_split_type", help="data_split_type",
                        default=None)
    parser.add_argument("--valid_split_pct", help="valid_split_pct",
                        type=float, default=0.15)
    parser.add_argument("--test_split_pct", help="test_split_pct",
                        type=float, default=0.2)
    parser.add_argument("--skip_frame_len", help="skip_frame_len",
                        type=int, default=1)
    parser.add_argument("--share_train_dataset", help="share_train_dataset",
                        action="store_true", default=False)

    # MIT_UCSD dataset specific properties
    parser.add_argument("--motion_type", help="motion_type",
                        default=None)
    parser.add_argument("--mit_ucsd_modality_features", help="mit_ucsd_modality_features",
                        default=None)

    # Optimization
    parser.add_argument("--lr_find", help="learning rate finder",
                        action="store_true", default=False)
    parser.add_argument("--lr_scheduler", help="lr_scheduler",
                        default=None)
    parser.add_argument("-cl", "--cycle_length", help="total number of executed iteration",
                        type=int, default=100)
    parser.add_argument("-cm", "--cycle_mul", help="total number of executed iteration",
                        type=int, default=2)

    # General Model Parameters
    parser.add_argument("--modality_encoder_type", help="encoder_type[mm_attn_encoder/gat_attn_encoder]",
                        default=None)

    # Unimodal Feature config
    parser.add_argument("--lstm_dropout", help="lstm_dropout",
                        type=float, default=0.1)
    parser.add_argument("--unimodal_attention_dropout", help="unimodal_attention_dropout",
                        type=float, default=0.1)
    parser.add_argument("-lld", "--lower_layer_dropout", help="lower layer dropout",
                        type=float, default=0.2)
    parser.add_argument("-uld", "--upper_layer_dropout", help="upper layer dropout",
                        type=float, default=0.2)

    # General Archi Config
    parser.add_argument("--layer_norm_type", help="layer_norm_type",
                        default='batch_norm')

    # Testing Config
    parser.add_argument("--test_models", help="test_models",
                        default='valid_loss,train_loss')
    parser.add_argument("--test_metrics", help="test_metrics",
                        default='loss,accuracy,f1_scores,precision,recall_scores')
    parser.add_argument("--is_test", help="evaluate on test dataset",
                        action="store_true", default=False)
    parser.add_argument("--only_testing", help="Perform only test on the pretrained model",
                        action="store_true", default=False)

    # Multi-task Config
    parser.add_argument("--task_name", help="task_name",
                        default=None)

    # Noisy training prop
    parser.add_argument("--train_noisy_sample_prob", help="train_noisy_sample_prob",
                        type=float, default=None)
    parser.add_argument("--valid_noisy_sample_prob", help="valid_noisy_sample_prob",
                        type=float, default=None)
    parser.add_argument("--test_noisy_sample_prob", help="test_noisy_sample_prob",
                        type=float, default=None)
    parser.add_argument("--noise_level", help="noise_level [json]",
                        default=None)
    parser.add_argument("--noise_type", help="noise_type [gaussian/random]",
                        default='random')
    parser.add_argument("--noisy_modalities", help="noisy_modalities",
                        default=None)

    # Student Training
    parser.add_argument("--is_student_training", help="Training student from a teacher model",
                        action="store_true", default=False)
    parser.add_argument("--is_student_training_alignment", help="Finetuning student from a teacher model",
                        action="store_true", default=False)
    parser.add_argument("--is_student_finetune", help="Finetuning student from a teacher model",
                        action="store_true", default=False)
    parser.add_argument("-rtcf", "--resume_teacher_checkpoint_filename", help="resume teacher model checkpoint filename",
                        default=None)
    parser.add_argument("-rscf", "--resume_student_checkpoint_filename", help="resume student model checkpoint filename",
                        default=None)
    parser.add_argument("--num_domains", help="num_domains",
                        type=int, default=4)
    parser.add_argument("--adversarial_loss_weight", help="adversarial_loss_weight",
                        type=float, default=0.5)
    parser.add_argument("--student_target_task_name", help="student_target_task_name",
                        default=None)
    parser.add_argument("--is_encoder_finetune", help="Finetuning encoder",
                        action="store_true", default=False)
    parser.add_argument("--task_list", help="task_list",
                        default=None)

    args = parser.parse_args()
    main(args=args)
