from itertools import cycle
import os
import sys
import json
import argparse
import time
from collections import defaultdict
import uuid
from tqdm import tqdm
import copy
from functools import partial

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

import numpy as np


from genrobo3d.train.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from genrobo3d.train.utils.save import ModelSaver, save_training_meta
from genrobo3d.train.utils.misc import NoOp, set_dropout, set_random_seed
from genrobo3d.train.utils.distributed import set_cuda, wrap_model, all_gather

from genrobo3d.train.optim import get_lr_sched, get_lr_sched_decay_rate
from genrobo3d.train.optim.misc import build_optimizer

from genrobo3d.configs.default import get_config

from genrobo3d.train.datasets.loader import build_dataloader
# from genrobo3d.train.datasets.simple_policy_dataset import (
#     SimplePolicyDataset, base_collate_fn, ptv3_collate_fn
# )
from genrobo3d.train.datasets.diffusion_policy_dataset import (
    SimplePolicyDataset, base_collate_fn, ptv3_collate_fn
)


from genrobo3d.models.simple_policy_ptv3 import (
    SimplePolicyPTV3AdaNorm, SimplePolicyPTV3CA, SimplePolicyPTV3Concat
)

import wandb
# Import hydra and omegaconf for Hydra-based config loading
import hydra
from omegaconf import DictConfig, OmegaConf


DATASET_FACTORY = {
    'SimplePolicyPTV3AdaNorm': (SimplePolicyDataset, ptv3_collate_fn),
    'SimplePolicyPTV3CA': (SimplePolicyDataset, ptv3_collate_fn),
    'SimplePolicyPTV3Concat': (SimplePolicyDataset, ptv3_collate_fn),
}

MODEL_FACTORY = {
    'SimplePolicyPTV3AdaNorm': SimplePolicyPTV3AdaNorm,
    'SimplePolicyPTV3CA': SimplePolicyPTV3CA,
    'SimplePolicyPTV3Concat': SimplePolicyPTV3Concat,
}


def main(config):
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)
    default_gpu, n_gpu, device = set_cuda(config)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )

    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    # load data training set
    dataset_class, dataset_collate_fn = DATASET_FACTORY[config.MODEL.model_class]
    trn_dataset = dataset_class(**config.TRAIN_DATASET, taskvars_filter=config.TRAIN.taskvars_filter)
    LOGGER.info(f'#num_train: {len(trn_dataset)}')
    trn_dataloader, pre_epoch = build_dataloader(
        trn_dataset, dataset_collate_fn, True, config
    )

    if config.VAL_DATASET.use_val:
        val_dataset = dataset_class(**config.VAL_DATASET, taskvars_filter=config.TRAIN.taskvars_filter)
        LOGGER.info(f"#num_val: {len(val_dataset)}")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.TRAIN.val_batch_size,
            num_workers=config.TRAIN.n_workers, pin_memory=True, collate_fn=dataset_collate_fn, sampler=RandomSampler(val_dataset, replacement=True)
        )
    else:
        val_dataloader = None

    
    val_loader = cycle(val_dataloader)  # Initial iterator
    LOGGER.info(f'#num_steps_per_epoch: {len(trn_dataloader)}')
    LOGGER.info(f"Validation Dataset Size: {len(val_dataset)}")
    LOGGER.info(f"Validation Batch Size: {config.TRAIN.val_batch_size}")

    LOGGER.info(f'#num_steps_per_epoch: {len(trn_dataloader)}')
    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(trn_dataloader) * config.TRAIN.num_epochs
    else:
        # assert config.TRAIN.num_epochs is None, 'cannot set num_train_steps and num_epochs at the same time.'
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(trn_dataloader)))
        
    if config.TRAIN.gradient_accumulation_steps > 1:
        config.TRAIN.num_train_steps *= config.TRAIN.gradient_accumulation_steps
        config.TRAIN.num_epochs *= config.TRAIN.gradient_accumulation_steps

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        if config.tfboard_log_dir is None:
            output_dir_tokens = config.output_dir.split('/')
            config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        TB_LOGGER.create(config.tfboard_log_dir)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    # Prepare model
    model_class = MODEL_FACTORY[config.MODEL.model_class]
    model = model_class(config.MODEL)
    # DDP: SyncBN
    if config.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if config.wandb_enable:
        wandb_dict = {}

    # Fix parameters
    if config.TRAIN.freeze_params.encoder:
        for param_name, param in model.named_parameters():
            if param_name.startswith('mae_encoder') and 'decoder_block' not in param_name:
                    param.requires_grad = False

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))
    
    OmegaConf.set_readonly(config, True)

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.TRAIN.resume_training:
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']
        restart_epoch = global_step // len(trn_dataloader)
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = 0
        global_step = restart_epoch * len(trn_dataloader) 

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                # TODO: mae_encoder.encoder.first_conv.0.weight
                if k == 'mae_encoder.encoder.first_conv.0.weight':
                    if v.size(1) != state_dict[k].size(1):
                        new_checkpoint[k] = torch.zeros_like(state_dict[k])
                        min_v_size = min(v.size(1), state_dict[k].size(1))
                        new_checkpoint[k][:, :min_v_size] = v[:, :min_v_size]
                if v.size() == state_dict[k].size():
                    if config.TRAIN.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
                        continue
                    new_checkpoint[k] = v
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)

    model.train()
    # set_dropout(model, config.TRAIN.dropout)
    model = wrap_model(model, device, config.local_rank, find_unused_parameters=True)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config.TRAIN)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    optimizer.zero_grad()
    optimizer.step()

    running_metrics = {}

    
    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        if global_step >= config.TRAIN.num_train_steps:
            break

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        pre_epoch(epoch_id)
        
        for step, batch in enumerate(trn_dataloader):
            # forward pass
            _, losses = model(batch, compute_loss=True, compute_final_action=False)

            # backward pass
            if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
                losses['total'] = losses['total'] / config.TRAIN.gradient_accumulation_steps
            losses['total'].backward()

            for key, value in losses.items():
                if config.wandb_enable:
                    wandb_dict.update({f'train_loss_{key}': value.item()})
                TB_LOGGER.add_scalar(f'step/loss_{key}', value.item(), global_step)
                running_metrics.setdefault(f'loss_{key}', RunningMeter(f'loss_{key}'))
                running_metrics[f'loss_{key}'](value.item())

            # optimizer update and logging
            if (step + 1) % config.TRAIN.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-8)
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)
                if config.wandb_enable:
                    wandb_dict.update({'lr': lr_this_step, 'global_step': global_step})

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.step()

                # update model params
                if config.TRAIN.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.grad_norm
                    )
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                    if config.wandb_enable:
                        wandb_dict.update({'grad_norm': grad_norm})
                optimizer.step()
                optimizer.zero_grad()
                if step % config.TRAIN.bar_steps == 0:
                    pbar.update(config.TRAIN.bar_steps)

            if global_step % config.TRAIN.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
                LOGGER.info('===============================================')
                if config.wandb_enable:
                    wandb.log(wandb_dict)             

            if global_step % config.TRAIN.save_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

            if (val_dataloader is not None) and (global_step % config.TRAIN.val_steps == 0):
                val_metrics = validate(model, val_loader, config.TRAIN.val_batches)
                LOGGER.info(f'=================Validation=================')
                metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
                LOGGER.info(metric_str)
                if config.wandb_enable:
                    wandb.log(val_metrics)
                LOGGER.info('===============================================')
                model.train()

            if global_step >= config.TRAIN.num_train_steps:
                break

    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(
            f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

        val_metrics = validate(model, val_loader)
        LOGGER.info(f'=================Validation=================')
        metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
        LOGGER.info(metric_str)
        LOGGER.info('===============================================')



@torch.no_grad()
def validate(model, val_loader, num_batches_per_step=5):
    model.eval()
    
    pos_loss, rot_loss, open_loss, total_loss = 0, 0, 0, 0
    total_pos_l2_err, total_quat_l1_err = 0, 0
    total_open_acc = 0
    total_samples = 0  # Tracks total sample count across batches
    
    # Global accuracy tracking
    total_pos_acc_0_01 = 0
    total_rot_acc_0_025 = 0
    total_rot_acc_0_05 = 0

    # Per-task metric storage
    per_task_metrics = {}

    for _ in range(num_batches_per_step):
        try:
            batch = next(val_loader)
        except StopIteration:
            print("Validation dataset finished, which should not happen.")

        pred_action, loss = model(batch, compute_loss=True)
        pred_action = pred_action.cpu()

        batch_size = pred_action.size(0)
        total_samples += batch_size  # Track total number of samples processed

        # Open accuracy
        pred_open = torch.sigmoid(pred_action[..., -1]) > 0.5
        batch_open_acc = (pred_open == batch["gt_actions"][..., -1].cpu()).float().sum().item()
        total_open_acc += batch_open_acc

        # Position and quaternion errors
        pos_l2 = ((pred_action[..., :3] - batch["gt_actions"][..., :3].cpu()) ** 2).sum(-1).sqrt()
        quat_l1 = (pred_action[..., 3:7] - batch["gt_quaternion"][..., :].cpu()).abs().sum(-1)
        quat_l1_ = (pred_action[..., 3:7] + batch["gt_quaternion"][..., :].cpu()).abs().sum(-1)
        quat_l1 = torch.min(quat_l1, quat_l1_)

        # Threshold-based accuracy (0 or 1)
        pos_acc_0_01 = (pos_l2 < 0.01).float().sum().item()
        rot_acc_0_025 = (quat_l1 < 0.025).float().sum().item()
        rot_acc_0_05 = (quat_l1 < 0.05).float().sum().item()

        # Accumulate total errors
        total_pos_l2_err += pos_l2.sum().item()
        total_quat_l1_err += quat_l1.sum().item()

        # Accumulate total accuracy metrics
        total_pos_acc_0_01 += pos_acc_0_01
        total_rot_acc_0_025 += rot_acc_0_025
        total_rot_acc_0_05 += rot_acc_0_05

        # Loss Accumulation
        if "layer_11" in loss:
            pos_loss += loss["layer_11_pos"].item()
            total_loss += loss["layer_11"].item()
        else:
            pos_loss += loss["pos"].item()
            rot_loss += loss["rot"].item()
            open_loss += loss["open"].item()
            total_loss += loss["total"].item()

        # Extract task names
        tasks = batch["data_ids"]
        task_names = [task.split("_peract")[0] for task in tasks]

        # Compute Per-Task Metrics
        for task_name in np.unique(task_names):
            task_mask = np.array(task_names) == task_name
            task_count = task_mask.sum()  # Count samples for this task in batch

            if task_name not in per_task_metrics:
                per_task_metrics[task_name] = {
                    "pos_l2_err_sum": 0, "quat_l1_err_sum": 0,
                    "pos_acc_0.01_sum": 0, "rot_acc_0.025_sum": 0, "rot_acc_0.05_sum": 0,
                    "open_acc_sum": 0, "count": 0
                }

            # Store per-task errors and accuracy (accumulate sums and counts)
            per_task_metrics[task_name]["pos_l2_err_sum"] += pos_l2[task_mask].sum().item()
            per_task_metrics[task_name]["quat_l1_err_sum"] += quat_l1[task_mask].sum().item()
            per_task_metrics[task_name]["pos_acc_0.01_sum"] += (pos_l2[task_mask] < 0.01).float().sum().item()
            per_task_metrics[task_name]["rot_acc_0.025_sum"] += (quat_l1[task_mask] < 0.025).float().sum().item()
            per_task_metrics[task_name]["rot_acc_0.05_sum"] += (quat_l1[task_mask] < 0.05).float().sum().item()
            per_task_metrics[task_name]["open_acc_sum"] += (pred_open[task_mask] == batch["gt_actions"][task_mask, -1].cpu()).float().sum().item()
            per_task_metrics[task_name]["count"] += task_count

    # Compute final averages
    total_samples = max(total_samples, 1)

    metrics = {
        "total/loss": total_loss / num_batches_per_step,
        "total/pos_loss": pos_loss / num_batches_per_step,
        "total/rot_loss": rot_loss / num_batches_per_step,
        "total/open_loss": open_loss / num_batches_per_step,

        # Total errors (mean per example)
        "total/pos_l2_err": total_pos_l2_err / total_samples,
        "total/quat_l1_err": total_quat_l1_err / total_samples,

        # Total accuracy
        "total/open_acc": total_open_acc / total_samples,
        "total/pos_acc_0.01": total_pos_acc_0_01 / total_samples,
        "total/rot_acc_0.025": total_rot_acc_0_025 / total_samples,
        "total/rot_acc_0.05": total_rot_acc_0_05 / total_samples,
    }

    # Store per-task metrics
    for task_name, task_data in per_task_metrics.items():
        count = max(task_data["count"], 1)  # Prevent division by zero
        metrics[f"per_task/{task_name}_pos_l2_err"] = task_data["pos_l2_err_sum"] / count
        metrics[f"per_task/{task_name}_quat_l1_err"] = task_data["quat_l1_err_sum"] / count
        metrics[f"per_task/{task_name}_pos_acc_0.01"] = task_data["pos_acc_0.01_sum"] / count
        metrics[f"per_task/{task_name}_rot_acc_0.025"] = task_data["rot_acc_0.025_sum"] / count
        metrics[f"per_task/{task_name}_rot_acc_0.05"] = task_data["rot_acc_0.05_sum"] / count
        metrics[f"per_task/{task_name}_open_acc"] = task_data["open_acc_sum"] / count

    return metrics





def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config


# Remove the build_args function and use Hydra for config loading.
@hydra.main(config_path="./", config_name="modified_ptv3")
def hydra_main(config: DictConfig):
    if config.wandb_enable:
        # gerenate a id including date and time
        time_id = f"{config.MODEL.model_class}_{time.strftime('%m%d-%H')}"
        # gnerate a UUID incase of same time_id
        wandb.init(project='pt3-diff', name=config.wandb_name + f"{time_id}_{str(uuid.uuid4())[:8]}", config=OmegaConf.to_container(config, resolve=True))
        main(config)
        wandb.finish()

    else:
        print(OmegaConf.to_container(config, resolve=True))
        main(config)

if __name__ == '__main__':
    hydra_main()
