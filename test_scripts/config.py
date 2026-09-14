from typing import Literal
from pydantic_settings import BaseSettings


class Config(BaseSettings, cli_parse_args=True):
    model_name: Literal[
        "resnet56", "resnet50", "resnet20", "vgg7bn", "vit", "deit", "pvt", "swin"
    ] = "resnet56"
    dataset: Literal["cifar10", "imagenet", "imagenet_small"] = "cifar10"
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
