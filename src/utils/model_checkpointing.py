import os
import numpy as np
import torch 

class ModelCheckpointing:
    def __init__(self,
                ckpt_base_dir,
                ckpt_filename,
                metrics,
                metrics_save_ckpt_mode,
                metrics_mode_dict,
                logger=None):
        self.ckpt_base_dir = ckpt_base_dir
        self.ckpt_filename = ckpt_filename
        self.metrics = metrics
        self.metrics_mode_dict = metrics_mode_dict 
        self.metrics_save_ckpt_mode = metrics_save_ckpt_mode
        self.logger = logger

        self.best_metrics_score = {}
        for metric in self.metrics:
            tm_init_value = np.Inf if self.metrics_mode_dict[metric]=='min' else 0
            self.best_metrics_score[metric] = tm_init_value
    
    def update_metric_save_ckpt(self, results, model, epoch, trainer=None):

        # save checkpoints and logging
        # epoch = results['epoch']
        for metric in self.metrics:
            if (metric not in results) or (metric not in self.metrics_save_ckpt_mode):
                continue
            if self._is_best_score(metric, results[metric]):
                if self.metrics_save_ckpt_mode[metric]:
                    ckpt = {'state_dict': model.state_dict(),
                        'stat': str(results)}
                    ckpt_filepath = f'{self.ckpt_base_dir}/best_{metric}_{self.ckpt_filename}'
                    self._save_model(ckpt_filepath, ckpt, trainer)
                    self.logger.log(f'Model save to {ckpt_filepath}')
                
                if self.logger is not None:
                    log_txt = f'\n###>>> Epoch {epoch}:- {metric} updated:'
                    comma = ''
                    for tm_metric in self.metrics:
                        if tm_metric in results:
                            tm_txt = '({:.5f} --> {:.5f})'.format(self.best_metrics_score[tm_metric], results[tm_metric])
                            log_txt = f'{log_txt}{comma} {tm_metric} {tm_txt}'
                            comma = ','
                    log_txt = f'{log_txt}\n'
                    self.logger.log(log_txt)
        
        # update the best metric scores
        for metric in self.metrics:
            if metric not in results:
                continue
            if self._is_best_score(metric, results[metric]):
                self.best_metrics_score[metric]=results[metric]
    
    def _save_model(self, ckpt_filepath, checkpoint, trainer):
        os.makedirs(os.path.dirname(ckpt_filepath), exist_ok=True)
        if trainer is not None:
            trainer.save_checkpoint(ckpt_filepath)
        else:
            torch.save(checkpoint, ckpt_filepath)

    def _is_best_score(self, metric, result):
        if self.metrics_mode_dict[metric]=='min':
            if self.best_metrics_score[metric]>result:
                return True
            else:
                return False
        else:
            if self.best_metrics_score[metric]<result:
                return True
            else:
                return False


    