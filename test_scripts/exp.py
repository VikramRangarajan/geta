import os
import sys
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile
from pydantic_settings import BaseSettings
from typing import Callable
from test_scripts.config import Config

import dotenv

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
EXPERIMENTS: dict[int | str, Callable] = {}


class ExperimentSettings(
    BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True
):
    cluster: str
    gpu: str
    gpus: int = 1
    exp: int


exp_settings = ExperimentSettings()

SCHOLAR_PREFIX = """#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -A scholar
#SBATCH -p gpu
#SBATCH --constraint={CONSTRAINT}
#SBATCH --mem=128G
#SBATCH --gres=gpu:{GPUS}
#SBATCH --signal=B:TERM@300
#SBATCH --job-name={JOBNAME}
#SBATCH --output=logs/%x-%j.out

cd {PROJECT_ROOT}
uv run accelerate launch {ACCELERATE_ARGS} test_scripts/Qtest_clean.py {ARGS}
"""

scholar_gpus = {"V100": "G", "A30": "H", "A40": "J"}

GILBRETH_PREFIX = """#!/bin/bash
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t {HOURS}:00:00
#SBATCH -q {QOS}
#SBATCH -p {PARTITION}
#SBATCH --mem=256G
#SBATCH --gres=gpu:{GPUS}
#SBATCH --signal=B:TERM@300
#SBATCH --job-name={JOBNAME}
#SBATCH --output=logs/%x-%j.out

cd {PROJECT_ROOT}
uv run accelerate launch {ACCELERATE_ARGS} test_scripts/Qtest_clean.py {ARGS}
"""

LOCAL_CMD = """
cd {PROJECT_ROOT}
uv run accelerate launch {ACCELERATE_ARGS} test_scripts/Qtest_clean.py {ARGS}
"""

PREFIXES = {"scholar": SCHOLAR_PREFIX, "gilbreth": GILBRETH_PREFIX, "local": LOCAL_CMD}


def exp(exp_num: int):
    def _func(f: Callable[[], None]):
        f_name = f.__name__  # type: ignore
        if exp_num in EXPERIMENTS or f_name in EXPERIMENTS:
            raise ValueError(f"Duplicate experiment: {exp_num} ({f_name})")

        def _print() -> None:
            print(f"Launching Experiment {f_name} ({exp_num})")
            f()

        EXPERIMENTS[exp_num] = _print
        EXPERIMENTS[str(exp_num)] = _print
        EXPERIMENTS[f_name] = _print
        return _print

    return _func


def dict_to_flags(flags: dict):
    flags_list = [f"--{k}={v}" for k, v in flags.items()]
    return " ".join(flags_list)


def submit_job(
    flags: str | dict,
    name: str,
    hours: int = 4,
    gpus: int = 1,
    nodes: int = 1,
    accelerate_args: str = "",
):
    if isinstance(flags, dict):
        flags = dict_to_flags(flags)
    constraint = None

    if exp_settings.cluster == "scholar":
        constraint = scholar_gpus[exp_settings.gpu.upper()]
        qos = "normal"
    elif exp_settings.cluster == "gilbreth":
        qos = "normal" if hours > 4 else "standby"
        if exp_settings.gpu.lower() not in ("a100", "a10", "a100-80gb"):
            raise ValueError(f"Invalid GPU: {exp_settings.gpu}")
        partition = {"a100": "a100-80gb", "a100-80gb": "a100-80gb", "a10": "a10"}[exp_settings.gpu.lower()]
    else:
        raise ValueError()
    slurm_script = PREFIXES[exp_settings.cluster].format(
        NODES=nodes,
        HOURS=hours,
        JOBNAME=name,
        QOS=qos,
        PARTITION=partition,
        GPUS=gpus,
        PROJECT_ROOT=PROJECT_ROOT,
        ARGS=flags,
        CONSTRAINT=constraint,
        ACCELERATE_ARGS=accelerate_args,
    )
    if exp_settings.cluster == "local":
        print("Running locally")
        run(["bash", "-c", slurm_script])
        return
    with NamedTemporaryFile(suffix=f"{name}.sh", delete=False) as f:
        pass
    print("Writing to", f.name)
    Path(f.name).write_text(slurm_script)
    run(["sbatch", f.name])


@exp(1)
def resnet20_cifar10_baseline():
    exp_name = "resnet20-cifar10-bf16"
    cfg = Config(
        model_name="resnet20",
        dataset="cifar10",
        batch_size=64,
        epochs=350,
        lr=1e-1,
        lr_quant=1e-4,
        weight_decay=1e-4,
        sparsity=0.35,
        projection_start_step=0,
        projection_periods=7,
        projection_steps=35,
        pruning_start_step=35,
        pruning_periods=5,
        pruning_steps=30,
        variant="sgd",
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        seed=0,
        output_dir="outputs/table2",
        data_dir="data/",
        ablation=exp_name,
        _cli_parse_args=False,
    )
    submit_job(
        dict_to_flags(cfg.model_dump()),
        exp_name,
        accelerate_args="--mixed_precision=bf16",
    )


@exp(2)
def resnet50_imagenet():
    assert exp_settings.cluster == "gilbreth" and "a100" in exp_settings.gpu
    for sparsity in 40, 50:
        exp_name = f"resnet50-imagenet1k-sp{sparsity}"
        cfg = Config(
            model_name="resnet50",
            dataset="imagenet",
            batch_size=64,
            epochs=120,
            lr=1e-1,
            lr_quant=1e-4,
            weight_decay=1e-4,
            sparsity=sparsity / 100,
            projection_start_step=5,
            projection_periods=5,
            projection_steps=5,
            pruning_start_step=10,
            pruning_periods=10,
            pruning_steps=10,
            variant="sgd",
            bit_reduction=2,
            min_bit_wt=4,
            max_bit_wt=16,
            seed=0,
            output_dir=f"outputs/table5sp{sparsity}",
            data_dir="data/",
            ablation=exp_name,
            _cli_parse_args=False,
        )
        submit_job(
            dict_to_flags(cfg.model_dump()),
            exp_name,
            accelerate_args="--mixed_precision=bf16",
        )


def main():
    exp = exp_settings.exp
    if exp not in EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment: got {exp}, expected one of {list(EXPERIMENTS.keys())}"
        )
    EXPERIMENTS[exp]()


if __name__ == "__main__":
    main()
