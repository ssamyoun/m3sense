import torch
import pytorch_lightning as pl

def get_pl_metrics(metric, num_classes):
    if metric=='accuracy':
        return pl.metrics.Accuracy()
        # return Accuracy()
    elif metric=='f1_scores':
        return pl.metrics.F1(num_classes)
    elif metric=='precision':
        return pl.metrics.Precision(num_classes)
    elif metric=='recall_scores':
        return pl.metrics.Recall(num_classes)


def cal_metrics(outputs, metrics, pl_metrics, stage_tag, trainer, device, is_multi_optimizers=False):
    results = {}
    for metric in metrics:
        metric_key = f'{stage_tag}_{metric}'
        results[f'epoch_{metric_key}'] = pl_metrics[metric_key].compute()

    if(is_multi_optimizers and type(outputs[0])==list):
        losses = [output[0]['loss'] for output in outputs]
        loss = torch.mean(torch.tensor(losses, device=device))
        torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
        loss = loss / trainer.world_size
        results[f'epoch_{stage_tag}_loss'] = loss

        losses_adv = [output[1]['loss'] for output in outputs]
        loss_adv = torch.mean(torch.tensor(losses, device=device))
        torch.distributed.all_reduce(loss_adv, op = torch.distributed.ReduceOp.SUM)
        loss_adv = loss_adv / trainer.world_size
        results[f'epoch_{stage_tag}_loss_adv'] = loss_adv
    else:
        losses = [output['loss'] for output in outputs]
        loss = torch.mean(torch.tensor(losses, device=device))
        torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
        loss = loss / trainer.world_size
        results[f'epoch_{stage_tag}_loss'] = loss
    return results

def cal_metrics_mt(outputs, metrics, pl_metrics, stage_tag, trainer, device, task_list, is_multi_optimizers=False):
    results = {}
    for task_name in task_list:
        for metric in metrics:
            metric_key = f'{stage_tag}_{metric}_{task_name}'
            results[f'epoch_{metric_key}'] = pl_metrics[metric_key].compute()

    if(is_multi_optimizers and type(outputs[0])==list):
        losses = [output[0]['loss'] for output in outputs]
        loss = torch.mean(torch.tensor(losses, device=device))
        torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
        loss = loss / trainer.world_size
        results[f'epoch_{stage_tag}_loss'] = loss

        losses_adv = [output[1]['loss'] for output in outputs]
        loss_adv = torch.mean(torch.tensor(losses, device=device))
        torch.distributed.all_reduce(loss_adv, op = torch.distributed.ReduceOp.SUM)
        loss_adv = loss_adv / trainer.world_size
        results[f'epoch_{stage_tag}_loss_adv'] = loss_adv
    else:
        losses = [output['loss'] for output in outputs]
        loss = torch.mean(torch.tensor(losses, device=device))
        torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
        loss = loss / trainer.world_size
        results[f'epoch_{stage_tag}_loss'] = loss
    return results

# def cal_metrics(outputs, metrics, pl_me trics, task_list, stage_tag, trainer, device):
#     results = {}
#     for metric in metrics:
#         temp_metric = []
#         for task in task_list:
#             metric_key = f'{task}_{stage_tag}_{metric}'
#             temp_metric.append(pl_metrics[metric_key].compute())
        
#         metric_key = f'{stage_tag}_{metric}'
#         results[f'epoch_{metric_key}'] = torch.mean(torch.tensor(temp_metric))
    
#     losses = [output['loss'] for output in outputs]
#     loss = torch.mean(torch.tensor(losses, device=device))
#     torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
#     loss = loss / trainer.world_size
#     results[f'epoch_{stage_tag}_loss'] = loss
#     return results

# from typing import Any, Callable, Optional

# import torch

# from pytorch_lightning.metrics.metric import Metric
# from pytorch_lightning.metrics.utils import _input_format_classification



# class Accuracy(Metric):
    
#     def __init__(
#         self,
#         threshold: float = 0.5,
#         compute_on_step: bool = True,
#         dist_sync_on_step: bool = False,
#         process_group: Optional[Any] = None,
#         dist_sync_fn: Callable = None,
#     ):
#         super().__init__(
#             compute_on_step=compute_on_step,
#             dist_sync_on_step=dist_sync_on_step,
#             process_group=process_group,
#             dist_sync_fn=dist_sync_fn,
#         )

#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#         self.threshold = threshold


#     def update(self, preds: torch.Tensor, target: torch.Tensor, weight:torch.LongTensor=1):
        
#         preds, target = _input_format_classification(preds, target, self.threshold)
#         assert preds.shape == target.shape

#         self.correct += torch.sum(preds == target) * weight
#         self.total += target.numel()



#     def compute(self):
#         return self.correct.float() / self.total
