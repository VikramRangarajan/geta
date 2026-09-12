from datasets import load_dataset
from pydantic import BaseModel
import json
import math
import os
from textwrap import dedent
from dataclasses import asdict
import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pydantic_settings import BaseSettings

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from sanity_check.backends.vgg7 import vgg7_bn
from test_scripts.geta_common import (
    resolve_data_dir,
    resolve_output_dir,
)
from utils.utils import check_accuracy_hf

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


def get_data_loader(
    dataset: str, batch_size: int, num_workers: int, prefetch_factor: int, data_dir=None
):
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
    elif dataset == "imagenet":
        input_size = (1, 3, 224, 224)
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        def get_transform_fn(transforms):
            def apply_transforms(example):
                return {
                    "image": transforms(example["image"]),
                    "label": example["label"],
                }

        train = load_dataset("ILSVRC/imagenet-1k", split="train")
        trainset = train.map(get_transform_fn(transform_train))
        test = load_dataset("ILSVRC/imagenet-1k", split="test")
        testset = test.map(get_transform_fn(transform_test))
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
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


class TrainingState(BaseModel):
    start_epoch: int
    best_acc1: float
    best_epoch: int

    def state_dict(self):
        return self.model_dump()

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)


def main(config: "Config"):
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        mode="default",  # Options: "default", "reduce-overhead", "max-autotune"
        fullgraph=True,
        dynamic=False,
    )
    accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
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
    output_logger.info(config.model_dump_json(indent=2))

    torch.manual_seed(config.seed)
    device = accelerator.device

    train_loader, test_loader, input_size = get_data_loader(
        config.dataset,
        config.batch_size,
        config.num_workers,
        config.prefetch_factor,
        data_dir,
    )
    num_classes = 10 if config.dataset == "cifar10" else 1000
    dummy_input = torch.rand(input_size).to(device)

    if config.model_name == "vgg7bn":
        model = vgg7_bn(num_classes=num_classes)
    elif config.model_name == "resnet20":
        model = resnet20_cifar10()
    elif config.model_name == "resnet56":
        model = resnet56_cifar10()
    elif config.model_name == "vit":
        from sanity_check.backends.vision_transformer.vision_transformer import (
            vit_small_patch16_224,
        )

        model = vit_small_patch16_224(pretrained=True, num_classes=1000)
    elif config.model_name == "deit":
        from sanity_check.backends.vision_transformer.DeiT import deit_tiny_patch16_224

        model = deit_tiny_patch16_224(pretrained=True, num_classes=1000)
    elif config.model_name == "pvt":
        from sanity_check.backends.vision_transformer.PVT import pvt_v2_b0

        model = pvt_v2_b0(pretrained=True, num_classes=1000)
    elif config.model_name == "swin":
        from sanity_check.backends.vision_transformer.Swin import (
            swin_tiny_patch4_window7_224,
        )

        model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=1000)

    model = model_to_quantize_model(model, num_bits=config.init_bit)

    oto = OTO(model.to(device), dummy_input=dummy_input)

    if config.model_name == "vit" or config.model_name == "deit":
        oto.mark_unprunable_by_param_names(["patch_embed.proj.weight", "pos_embed"])
    elif config.model_name == "pvt":
        model = None
    elif config.model_name == "swin":
        unprunable_list = ["patch_embed.proj.weight", "pos_embed"]
        for name, param in model.named_parameters():
            if "attn.qkv." in name:
                unprunable_list.append(name)

        oto.mark_unprunable_by_param_names(unprunable_list)

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
    else:
        raise NotImplementedError()

    # Get full/original floating-point model MACs, BOPs, and number of parameters
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)
    full_weight_size = oto.compute_weight_size(in_million=True)
    full_average_bit_width = oto.compute_average_bit_width()

    # Hotfix for full_bops calculation
    full_bops["total"] = full_bops["total"] * 32 / config.init_bit

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

    state = TrainingState(start_epoch=0, best_acc1=0.0, best_epoch=0)

    accelerator.register_for_checkpointing(state)

    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )

    # Checkpoint resume: check TRAINER_RESUME / SLURM_RESTART_COUNT or existing checkpoint
    if (
        os.environ.get("TRAINER_RESUME") == "1"
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        accelerator.load_state()
    for epoch in range(state.start_epoch, config.epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        ):
            with accelerator.accumulate(model):
                if config.dataset == "imagenet":
                    inputs, targets = batch["image"], batch["labels"]
                else:
                    inputs, targets = batch

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

                with accelerator.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    running_loss += loss.detach()
        running_loss = running_loss.item()
        opt_metrics = optimizer.optimizer.compute_metrics()
        running_loss_avg = running_loss / len(train_loader)

        accuracy1, accuracy5 = check_accuracy_hf(model, accelerator, test_loader, two_input=False)
        if accelerator.is_main_process:
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
        if accuracy1 > state.best_acc1:
            state.best_acc1 = accuracy1
            best_epoch = epoch
        # Save checkpoint for resume (every epoch)
        accelerator.save_state(os.path.join(checkpoint_dir, wandb.run.id))

    # Construct the subnet and get the compressed model
    if accelerator.is_main_process:
        output_logger.info(f"Best epoch: {best_epoch}. Best acc1: {state.best_acc1}%")
        output_logger.info("Training completed. Constructing subnet...")
        oto.construct_subnet(out_dir=os.path.join(output_dir, "subnet"))
        compressed_model = torch.load(oto.compressed_model_path)
        oto_compressed = OTO(compressed_model, dummy_input)

        msg = dedent(f"""
        Full MACs for Q{config.model_name}: {full_macs["total"]} M MACs
        Full BOPs for Q{config.model_name}: {full_bops["total"]} M BOPs
        Full num params for Q{config.model_name}: {full_num_params} M params
        Full weight size for Q{config.model_name}: {full_weight_size["total"]} MB
        Full average weight bit width for Q{config.model_name}: {full_average_bit_width} bits
        """)
        output_logger.info(msg)
        if "layer_info" in full_macs and "layer_info" in full_bops:
            output_logger.info("Layer-by-layer breakdown for full model:")
            output_logger.info(
                f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}"
            )
            output_logger.info("-" * 75)
            for mac_info, bop_info in zip(
                full_macs["layer_info"], full_bops["layer_info"]
            ):
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

        msg = dedent(f"""
            Compressed MACs for Q{config.model_name}: {compressed_macs["total"]} M MACs
            Compressed BOPs for Q{config.model_name}: {compressed_bops["total"]} M BOPs
            Compressed num params for Q{config.model_name}: {compressed_num_params} M params
            Compressed weight size for Q{config.model_name}: {compressed_weight_size["total"]} MB
            Compressed average weight bit width for Q{config.model_name}: {compressed_average_bit_width} bits
        """)
        output_logger.info(msg)
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

    msg = dedent(f"""
        MAC reduction    : {(1.0 - compressed_macs["total"] / full_macs["total"]) * 100}%
        BOP reduction    : {(1.0 - compressed_bops["total"] / full_bops["total"]) * 100}%
        Param reduction  : {(1.0 - compressed_num_params / full_num_params) * 100}%
        MAC ratio: {full_macs["total"] / compressed_macs["total"]}
        BOP compresion ratio: {full_bops["total"] / compressed_bops["total"]}
        """)
    output_logger.info(msg)

    full_model_size = os.path.getsize(oto.full_group_sparse_model_path) / (1024**3)
    compressed_model_size = os.path.getsize(oto.compressed_model_path) / (1024**3)
    output_logger.info(f"Size of full/ model: {full_model_size:.4f} GB")
    output_logger.info(f"Size of compressed model: {compressed_model_size:.4f} GB")

    if accelerator.is_main_process:
        wandb.log(
            {
                "MAC_reduction": (1.0 - compressed_macs["total"] / full_macs["total"])
                * 100,
                "BOP_reduction": (1.0 - compressed_bops["total"] / full_bops["total"])
                * 100,
                "Param_reduction": (1.0 - compressed_num_params / full_num_params)
                * 100,
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
    model_name: Literal[
        "resnet56", "resnet20", "vgg7bn", "vit", "deit", "pvt", "swin"
    ] = "resnet56"
    dataset: Literal["cifar10", "imagenet"] = "cifar10"
    batch_size: int = 64
    epochs: int = 1
    lr: float = 1e-1  # 1e-3 for imagenet
    lr_quant: float = 1e-3  # 1e-4 for imagenet
    weight_decay: float = 1e-4
    sparsity: float = 0.4  # 0.3 for imagenet
    projection_start_step: int = 10  # 5 for imagenet
    projection_periods: int = 5
    pruning_start_step: int = 20  # 10 for imagenet
    pruning_periods: int = 10
    projection_steps: int = 10  # 5 for imagenet
    pruning_steps: int = 30  # 20 for imagenet
    lr_step: int = 100
    lr_gamma: float = 0.1
    variant: str = "sgd"  # adam for imagenet
    bit_reduction: int = 2
    init_bit: int = 16
    min_bit_wt: int = 4
    max_bit_wt: int = 16
    min_bit_act: int = 4  # 2 for imagenet
    max_bit_act: int = 6  # 16 for imagenet
    mix_up: bool = False
    label_smooth: bool = False
    seed: int = 0
    ablation: str = "qhesso"
    output_dir: str | None = None
    data_dir: str | None = None

    # Dataloader params
    num_workers: int = 8
    prefetch_factor: int = 2


def get_config():

    config = Config()

    # scale lr wd with batch size
    if config.batch_size != 64 and config.dataset == "cifar10":
        config.lr *= config.batch_size / 64
        config.weight_decay *= config.batch_size / 64

    assert (
        config.pruning_start_step
        == config.projection_start_step + config.projection_steps
    )
    return config


if __name__ == "__main__":
    main(get_config())
