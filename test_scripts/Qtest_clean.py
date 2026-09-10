import json
import math
import os
import warnings
from dataclasses import asdict
import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pydantic_settings import BaseSettings
from torch import nn

# from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# from transformers import AutoImageProcessor
from only_train_once import OTO
from only_train_once.optimizer.utils import (
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from sanity_check.backends.vgg7 import vgg7_bn
from test_scripts.geta_common import (
    resolve_data_dir,
    resolve_output_dir,
)
from utils.utils import check_accuracy

logging.basicConfig(level=logging.INFO, format="%(message)s")
output_logger = logging.getLogger(__name__)


def get_quant_param_dict(model):
    # Access quantization parameter information
    param_dict = {}
    for name, param in model.named_parameters():
        if "d_quant" in name or "t_quant" in name or "q_m" in name:
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]
            if layer_name in param_dict:
                param_dict[layer_name][param_name] = param.item()
            else:
                param_dict[layer_name] = {}
                param_dict[layer_name][param_name] = param.item()
    return param_dict


def get_bitwidth_dict(param_dict):
    bit_dict = {}

    for key in param_dict.keys():
        bit_dict[key] = {}

        d_quant_wt = param_dict[key]["d_quant_wt"]
        q_m_wt = abs(param_dict[key]["q_m_wt"])
        if "t_quant_wt" in param_dict[key]:
            t_quant_wt = param_dict[key]["t_quant_wt"]
        else:
            t_quant_wt = 1.0
        bit_width_wt = (
            math.log2(math.exp(t_quant_wt * math.log(q_m_wt)) / abs(d_quant_wt) + 1) + 1
        )
        bit_dict[key]["weight"] = bit_width_wt

        if "d_quant_act" in param_dict[key]:
            d_quant_act = param_dict[key]["d_quant_act"]
            q_m_act = abs(param_dict[key]["q_m_act"])
            if "t_quant_act" in param_dict[key]:
                t_quant_act = param_dict[key]["t_quant_act"]
            else:
                t_quant_act = 1.0
            bit_width_act = (
                math.log2(
                    math.exp(t_quant_act * math.log(q_m_act)) / abs(d_quant_act) + 1
                )
                + 1
            )
            bit_dict[key]["activation"] = bit_width_act

    return bit_dict


def compute_bop_compression_ratio(
    original,
    compressed,
    bitwidths,
    input_size=(1, 3, 32, 32),
    original_weight_bitwidth=32,
    activation_bitwidth=32,  # Fixed activation bitwidth for both models
    verbose=True,
):
    total_original_bop = 0
    total_compressed_bop = 0
    total_original_mac = 0
    total_compressed_mac = 0
    current_input_size = input_size
    prev_original_channels = input_size[1]
    prev_compressed_channels = input_size[1]
    prev_pl = 0  # Initial pruning ratio
    layer_idx = 0
    for (name, original_layer), (_, compressed_layer) in zip(
        original.named_modules(), compressed.named_modules()
    ):
        if isinstance(original_layer, (nn.Conv2d, nn.Linear)) and name in bitwidths:
            # Compute pruning ratios
            original_channels = (
                original_layer.out_channels
                if isinstance(original_layer, nn.Conv2d)
                else original_layer.out_features
            )
            compressed_channels = (
                compressed_layer.out_channels
                if isinstance(compressed_layer, nn.Conv2d)
                else compressed_layer.out_features
            )
            pl = 1 - (compressed_channels / original_channels)
            Pl = 1 - (1 - prev_pl) * (1 - pl)  # Layerwise pruning ratio

            # Compute dimensions
            if isinstance(original_layer, nn.Conv2d):
                mw_l, mh_l = current_input_size[2], current_input_size[3]
                kw, kh = original_layer.kernel_size
            else:  # Linear layer
                mw_l, mh_l = 1, 1
                kw, kh = 1, 1

            # Compute MAC counts
            mac_original = (
                (1 - prev_pl)
                * prev_original_channels
                * (1 - pl)
                * original_channels
                * mw_l
                * mh_l
                * kw
                * kh
            )
            mac_compressed = (
                (1 - prev_pl)
                * prev_compressed_channels
                * (1 - pl)
                * compressed_channels
                * mw_l
                * mh_l
                * kw
                * kh
            )

            # Add MAC counts to totals
            total_original_mac += mac_original
            total_compressed_mac += mac_compressed

            # Compute BOP counts
            bw_l = round(bitwidths[name])
            bop_original = mac_original * original_weight_bitwidth * activation_bitwidth
            bop_compressed = mac_compressed * bw_l * activation_bitwidth

            total_original_bop += bop_original
            total_compressed_bop += bop_compressed

            if verbose:
                output_logger.info(f"Layer name: {name}, Layer index: {layer_idx}")
                output_logger.info(
                    f"Original channels: {original_channels}, Compressed channels: {compressed_channels}"
                )
                output_logger.info(
                    f"Pruning ratio (pl): {pl:.4f}, Layerwise pruning ratio (Pl): {Pl:.4f}"
                )
                output_logger.info(
                    f"MAC count - Original: {mac_original / 1e6:.4f} M, Compressed: {mac_compressed / 1e6:.4f} M"
                )
                output_logger.info(
                    f"BOP count - Original: {bop_original / 1e9:.4f} G, Compressed: {bop_compressed / 1e9:.4f} G"
                )
                output_logger.info(
                    f"Weight Bitwidth - Original: {original_weight_bitwidth}, Compressed: {bw_l}"
                )
                output_logger.info(f"Activation Bitwidth: {activation_bitwidth}")
                output_logger.info("--------------------")

            # Update for next layer
            prev_original_channels = original_channels
            prev_compressed_channels = compressed_channels
            prev_pl = pl
            if isinstance(original_layer, nn.Conv2d):
                current_input_size = (
                    current_input_size[0],
                    original_channels,
                    (
                        current_input_size[2]
                        + 2 * original_layer.padding[0]
                        - original_layer.kernel_size[0]
                    )
                    // original_layer.stride[0]
                    + 1,
                    (
                        current_input_size[3]
                        + 2 * original_layer.padding[1]
                        - original_layer.kernel_size[1]
                    )
                    // original_layer.stride[1]
                    + 1,
                )
            layer_idx += 1

    bop_compression_ratio = (
        total_original_bop / total_compressed_bop
        if total_compressed_bop > 0
        else float("inf")
    )
    mac_compression_ratio = (
        total_original_mac / total_compressed_mac
        if total_compressed_mac > 0
        else float("inf")
    )

    if verbose:
        output_logger.info(f"Total Original MAC: {total_original_mac / 1e9:.4f} GMACs")
        output_logger.info(
            f"Total Compressed MAC: {total_compressed_mac / 1e9:.4f} GMACs"
        )
        output_logger.info(f"MAC Compression Ratio: {mac_compression_ratio:.4f}")
        output_logger.info(f"Total Original BOP: {total_original_bop / 1e9:.4f} GBOPs")
        output_logger.info(
            f"Total Compressed BOP: {total_compressed_bop / 1e9:.4f} GBOPs"
        )
        output_logger.info(f"BOP Compression Ratio: {bop_compression_ratio:.4f}")

    return bop_compression_ratio, total_original_mac, total_compressed_mac


def get_data_loader(dataset: str, batch_size: int, num_workers: int, data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    if dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        trainset = CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
        input_size = (1, 3, 32, 32)
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    elif dataset == "imagenet":
        raise ValueError("Unsupported dataset")
    else:
        raise ValueError("Unsupported dataset")

    return train_loader, test_loader, input_size


def one_hot(y, num_classes, smoothing_eps=None):
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y


def cross_entropy_onehot_target(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


def mixup_func(input, target, alpha=0.2):
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(
        1 - gamma, perm_target
    )


class WarmupThenScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        if not self.finished:
            self.after_scheduler.base_lrs = [
                base_lr * (self.warmup_steps + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
            self.finished = True
        return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_steps)
        else:
            return super().step(epoch)


def main(config: "Config"):

    assert (
        config.pruning_start_step
        == config.projection_start_step + config.projection_steps
    )

    wandb.init(config=config.model_dump())

    # Messaging logger
    output_dir = resolve_output_dir(
        config.output_dir, f"{config.model_name}_{config.variant}_{config.sparsity}"
    )
    data_dir = resolve_data_dir(config.data_dir)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup info
    output_logger.info(config)

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = 1
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    train_loader, test_loader, input_size = get_data_loader(
        config.dataset, config.batch_size * num_gpus, config.num_workers, data_dir
    )
    num_classes = 10 if config.dataset == "cifar10" else 1000
    dummy_input = torch.rand(input_size).to(device)

    if config.model_name == "vgg7bn":
        model = vgg7_bn(num_classes=num_classes)
        model = model_to_quantize_model(model)
    elif config.model_name == "resnet20":
        model = resnet20_cifar10()
        model = model_to_quantize_model(model)
    elif config.model_name == "resnet56":
        model = resnet56_cifar10()
        model = model_to_quantize_model(model)

    oto = OTO(model.to(device), dummy_input=dummy_input)

    # Add the visualization to make sure that everything quant_act_layers.py works well.
    # oto.visualize(view=False, out_dir='./cache', display_flops=True, display_params=True, display_macs=True)
    # exit()

    if config.ablation == "qhesso":
        pruning_periods = config.pruning_periods
        total_pruning_steps = 0
        if config.pruning_steps == 0:
            total_pruning_steps = 1
            pruning_periods = 1
        else:
            total_pruning_steps = config.pruning_steps * len(train_loader)
        optimizer = oto.geta(
            variant=config.variant,
            lr=config.lr,
            lr_quant=config.lr_quant,
            first_momentum=0.9,
            weight_decay=config.weight_decay,
            target_group_sparsity=config.sparsity,
            start_projection_step=config.projection_start_step * len(train_loader),
            projection_periods=config.projection_periods,
            projection_steps=config.projection_steps * len(train_loader),
            start_pruning_step=config.pruning_start_step * len(train_loader),
            pruning_periods=pruning_periods,
            pruning_steps=total_pruning_steps,  # pruning_steps * len(train_loader),
            bit_reduction=config.bit_reduction,
            min_bit_wt=config.min_bit_wt,
            max_bit_wt=config.max_bit_wt,
            min_bit_act=config.min_bit_act,
            max_bit_act=config.max_bit_act,
        )

    # Get full/original floating-point model MACs, BOPs, and number of parameters
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)
    full_weight_size = oto.compute_weight_size(in_million=True)
    full_average_bit_width = oto.compute_average_bit_width()

    # Hotfix for full_bops calculation
    full_bops["total"] = full_bops["total"] * 32 / 16

    if not config.label_smooth:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = cross_entropy_onehot_target
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=lr_step*len(train_loader), gamma=lr_gamma
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader), eta_min=0
    )
    lr_scheduler = WarmupThenScheduler(
        optimizer, warmup_steps=5 * len(train_loader), after_scheduler=lr_scheduler
    )
    if num_gpus > 1:
        output_logger.info(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)

    best_epoch = 0
    best_acc1 = 0.0
    loss_list = []
    start_epoch = 0
    # Checkpoint resume: check TRAINER_RESUME / SLURM_RESTART_COUNT or existing checkpoint
    if (
        os.environ.get("TRAINER_RESUME") == "1"
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        ckpt_path = scan_checkpoint(checkpoint_dir, "ckpt_")
        if ckpt_path is not None:
            try:
                output_logger.info(f"Attempting resume from {ckpt_path}")
                ckpt = load_checkpoint(ckpt_path, device)
                # model
                model_to_load = model.module if num_gpus > 1 else model
                model_to_load.load_state_dict(ckpt["model_state_dict"])
                # Restore optimizer counters without full load_state_dict to avoid ID mismatch
                opt_state = ckpt.get("optimizer_state_dict", {})
                for key in [
                    "num_steps",
                    "curr_pruning_period",
                    "start_pruning_step",
                    "pruning_periods",
                    "pruning_steps",
                    "start_projection_step",
                    "projection_periods",
                    "projection_steps",
                    "pruning_period_duration",
                    "projection_period_duration",
                    "target_num_redundant_groups",
                    "pruned_group_idxes",
                    "bit_layers",
                    "min_bit_wt",
                    "max_bit_wt",
                    "min_bit_act",
                    "max_bit_act",
                ]:
                    if key in opt_state:
                        try:
                            setattr(optimizer, key, opt_state[key])
                        except Exception:
                            pass
                # Restore per-group fields
                try:
                    ckpt_groups = opt_state.get("param_groups", [])
                    for pg, ckpt_pg in zip(optimizer.param_groups, ckpt_groups):
                        for k in [
                            "important_idxes",
                            "active_redundant_idxes",
                            "pruned_idxes",
                            "importance_scores",
                        ]:
                            if k in ckpt_pg:
                                pg[k] = ckpt_pg[k]
                except Exception as e:
                    output_logger.warning(f"Could not restore param_groups: {e}")
                # scheduler
                if (
                    "scheduler_state_dict" in ckpt
                    and ckpt["scheduler_state_dict"] is not None
                ):
                    try:
                        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    except Exception as e:
                        output_logger.warning(f"Could not load scheduler state: {e}")
                start_epoch = ckpt["epoch"] + 1
                best_acc1 = ckpt.get("best_acc1", 0.0)
                best_epoch = ckpt.get("best_epoch", 0)
                output_logger.info(
                    f"Resumed from epoch {start_epoch} (ckpt epoch {ckpt['epoch']}), best_acc1={best_acc1:.2f}%"
                )
            except Exception as e:
                output_logger.warning(
                    f"Failed to resume from checkpoint {ckpt_path}: {e}"
                )
                import traceback

                output_logger.warning(traceback.format_exc())
                start_epoch = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        ):
            if config.dataset == "imagenet":
                inputs, targets = batch["pixel_values"], batch["labels"]
            else:
                inputs, targets = batch
            inputs, targets = (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )

            with torch.no_grad():
                if config.label_smooth and not config.mix_up:
                    targets = one_hot(
                        targets, num_classes=num_classes, smoothing_eps=0.1
                    )
                if not config.label_smooth and config.mix_up:
                    targets = one_hot(targets, num_classes=num_classes)
                    inputs, targets = mixup_func(inputs, targets)
                if config.mix_up and config.label_smooth:
                    targets = one_hot(
                        targets, num_classes=num_classes, smoothing_eps=0.1
                    )
                    inputs, targets = mixup_func(inputs, targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.grad_clipping()
            optimizer.step()
            running_loss += loss.item()
            lr_scheduler.step()

        opt_metrics = optimizer.compute_metrics()
        running_loss_avg = running_loss / len(train_loader)

        accuracy1, accuracy5 = check_accuracy(
            model.module if num_gpus > 1 else model, test_loader, two_input=False
        )
        avg_wt_bit = oto.compute_average_bit_width()
        output_logger.info(
            f"Epoch: {epoch}, loss: {running_loss_avg:5.3f}, norm_all: {opt_metrics.norm_params:5.2f}, grp_sparsity: {opt_metrics.group_sparsity:5.2f}, acc1: {accuracy1:5.2f}%, acc5: {accuracy5:5.2f}%, norm_import: {opt_metrics.norm_important_groups:5.2f}, norm_redund: {opt_metrics.norm_redundant_groups:5.2f}, num_grp_import: {opt_metrics.num_important_groups:5.2f}, num_grp_redund: {opt_metrics.num_redundant_groups:5.2f}, avg_wt_bit_width: {avg_wt_bit:5.2f}"
        )
        wandb.log(
            dict(
                epoch=epoch,
                running_loss_avg=running_loss_avg,
                accuracy1=accuracy1,
                accuracy5=accuracy5,
                avg_wt_bit_width=avg_wt_bit,
                lr=optimizer.param_groups[0]["lr"],
                **asdict(opt_metrics),
            )
        )
        if accuracy1 > best_acc1:
            best_acc1 = accuracy1
            best_epoch = epoch
            torch.save(model, os.path.join(log_dir, "resnet20_best_acc1.pt"))
        # Save checkpoint for resume (every epoch)
        try:
            ckpt = optimizer.create_checkpoint(
                model.module if num_gpus > 1 else model, epoch, running_loss_avg
            )
            ckpt["best_acc1"] = best_acc1
            ckpt["best_epoch"] = best_epoch
            try:
                ckpt["scheduler_state_dict"] = lr_scheduler.state_dict()
            except:
                ckpt["scheduler_state_dict"] = None
            save_checkpoint(os.path.join(checkpoint_dir, f"ckpt_{epoch}.pt"), ckpt)
            # keep only last 3 checkpoints
            ckpts = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_")],
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            for old in ckpts[:-3]:
                try:
                    os.remove(os.path.join(checkpoint_dir, old))
                except:
                    pass
        except Exception as e:
            output_logger.warning(f"Failed to save checkpoint at epoch {epoch}: {e}")

        # loss_list.append(running_loss_avg)

    output_logger.info(f"Best epoch: {best_epoch}. Best acc1: {best_acc1}%")
    output_logger.info("Training completed. Constructing subnet...")

    # Construct the subnet and get the compressed model
    oto.construct_subnet(out_dir=os.path.join(output_dir, "subnet"))
    compressed_model = torch.load(oto.compressed_model_path)
    oto_compressed = OTO(compressed_model, dummy_input)

    output_logger.info(
        f"Full MACs for Q{config.model_name}: {full_macs['total']} M MACs"
    )
    output_logger.info(
        f"Full BOPs for Q{config.model_name}: {full_bops['total']} M BOPs"
    )
    output_logger.info(
        f"Full num params for Q{config.model_name}: {full_num_params} M params"
    )
    output_logger.info(
        f"Full weight size for Q{config.model_name}: {full_weight_size['total']} MB"
    )
    output_logger.info(
        f"Full average weight bit width for Q{config.model_name}: {full_average_bit_width} bits"
    )
    if "layer_info" in full_macs and "layer_info" in full_bops:
        output_logger.info("Layer-by-layer breakdown for full model:")
        output_logger.info(
            f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}"
        )
        output_logger.info("-" * 75)
        for mac_info, bop_info in zip(full_macs["layer_info"], full_bops["layer_info"]):
            output_logger.info(
                f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
            )

    # Get compressed model MACs, BOPs, and number of parameters
    compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
    compressed_bops = oto_compressed.compute_bops(
        in_million=True, layerwise=True
    )  # we adjust the calculation to subtract 1 from the activation to simulate unsigned activations. (post relu)
    compressed_num_params = oto_compressed.compute_num_params(in_million=True)
    compressed_weight_size = oto_compressed.compute_weight_size(in_million=True)
    compressed_average_bit_width = oto_compressed.compute_average_bit_width()

    output_logger.info(
        f"Compressed MACs for Q{config.model_name}: {compressed_macs['total']} M MACs"
    )
    output_logger.info(
        f"Compressed BOPs for Q{config.model_name}: {compressed_bops['total']} M BOPs"
    )
    output_logger.info(
        f"Compressed num params for Q{config.model_name}: {compressed_num_params} M params"
    )
    output_logger.info(
        f"Compressed weight size for Q{config.model_name}: {compressed_weight_size['total']} MB"
    )
    output_logger.info(
        f"Compressed average weight bit width for Q{config.model_name}: {compressed_average_bit_width} bits"
    )

    wandb.log(
        {
            "full_macs": full_macs["total"],
            "full_bops": full_bops["total"],
            "full_num_params": full_num_params,
            "full_weight_size": full_weight_size["total"],
            "full_average_bit_width": full_average_bit_width,
            "compressed_macs": compressed_macs["total"],
            "compressed_bops": compressed_bops["total"],
            "compressed_num_params": compressed_num_params,
            "compressed_weight_size": compressed_weight_size["total"],
            "compressed_average_bit_width": compressed_average_bit_width,
        }
    )

    if "layer_info" in compressed_macs and "layer_info" in compressed_bops:
        output_logger.info("Layer-by-layer breakdown for compressed model:")
        output_logger.info(
            f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}"
        )
        output_logger.info("-" * 75)
        for mac_info, bop_info in zip(
            compressed_macs["layer_info"], compressed_bops["layer_info"]
        ):
            output_logger.info(
                f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
            )

    output_logger.info(
        f"MAC reduction    : {(1.0 - compressed_macs['total'] / full_macs['total']) * 100}%"
    )
    output_logger.info(
        f"BOP reduction    : {(1.0 - compressed_bops['total'] / full_bops['total']) * 100}%"
    )
    output_logger.info(
        f"Param reduction  : {(1.0 - compressed_num_params / full_num_params) * 100}%"
    )
    output_logger.info(f"MAC ratio: {full_macs['total'] / compressed_macs['total']}")
    output_logger.info(
        f"BOP compresion ratio: {full_bops['total'] / compressed_bops['total']}"
    )

    full_model_size = os.path.getsize(oto.full_group_sparse_model_path) / (1024**3)
    compressed_model_size = os.path.getsize(oto.compressed_model_path) / (1024**3)
    output_logger.info(f"Size of full/ model: {full_model_size:.4f} GB")
    output_logger.info(f"Size of compressed model: {compressed_model_size:.4f} GB")

    wandb.log(
        {
            "MAC_reduction": (1.0 - compressed_macs["total"] / full_macs["total"])
            * 100,
            "BOP_reduction": (1.0 - compressed_bops["total"] / full_bops["total"])
            * 100,
            "Param_reduction": (1.0 - compressed_num_params / full_num_params) * 100,
            "MAC_ratio": full_macs["total"] / compressed_macs["total"],
            "BOP_compression_ratio": full_bops["total"] / compressed_bops["total"],
            "full_model_size": full_model_size,
            "compressed_model_size": compressed_model_size,
        }
    )

    # Print and visualize each layer bit width info
    param_dict = get_quant_param_dict(model)
    bit_dict = get_bitwidth_dict(param_dict)
    output_logger.info("=========================")
    output_logger.info(json.dumps(bit_dict, indent=2))


class Config(BaseSettings, cli_parse_args=True):
    model_name: Literal["resnet56", "resnet20", "vgg7bn"] = "resnet56"
    dataset: Literal["cifar10", "imagenet"] = "cifar10"
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 1
    lr: float = 1e-1
    lr_quant: float = 1e-3
    weight_decay: float = 1e-4
    sparsity: float = 0.4
    projection_start_step: int = 10
    projection_periods: int = 5
    pruning_start_step: int = 20
    pruning_periods: int = 10
    projection_steps: int = 10
    pruning_steps: int = 30
    lr_step: int = 100
    lr_gamma: float = 0.1
    variant: str = "sgd"
    bit_reduction: int = 2
    min_bit_wt: int = 4
    max_bit_wt: int = 16
    min_bit_act: int = 4
    max_bit_act: int = 6
    mix_up: bool = False
    label_smooth: bool = False
    seed: int = 0
    ablation: str = "qhesso"
    output_dir: str | None = None
    data_dir: str | None = None


def get_config():

    config = Config()

    # scale lr wd with batch size
    if config.batch_size != 64:
        config.lr *= config.batch_size / 64
        config.weight_decay *= config.batch_size / 64
    return config


if __name__ == "__main__":
    main(get_config())
