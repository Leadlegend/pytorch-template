import hydra
import torch.optim as opt

from functools import partial
from dataclasses import dataclass
from typing import Optional, List, Union
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from model.criterions import *


cfg2opt = {
    "adam": partial(opt.Adam, betas=(0.9, 0.99), eps=1e-05),
    "sgd": opt.SGD,
}
cfg2sch = {
    "None":
    None,
    "Plateau":
    partial(
        opt.lr_scheduler.ReduceLROnPlateau,
        factor=0.9,
        mode='min',
        patience=9,
        cooldown=2,
        min_lr=2e-5,
    ),
}
cfg2ep_crt = {
    'none': None,
    "pearson": None,
    'spearman': None,
}


@dataclass
class DatasetConfig:
    path: Union[List[str], str]
    batch_size: int = 1
    shuffle: bool = False
    pin: bool = False
    workers: int = 0
    lazy: bool = False
    label: Optional[bool] = True


@dataclass
class DataConfig:
    train: Optional[DatasetConfig]
    val: Optional[DatasetConfig]
    test: Optional[DatasetConfig]


@dataclass
class TrainerConfig:
    lr: float  # learning rate
    epoch: int  # epoch number
    device: str  # Cuda / cpu
    save_dir: str  # model checkpoint saving directory
    save_period: int = 1  # save one checkpoint every $save_period epoch
    ckpt: Optional[str] = None  # model initialization
    optimizer: Optional[str] = 'adam'  # optimizer name
    scheduler: Optional[str] = 'None'  # lr_scheduler name
    epoch_criterion: Optional[str] = 'none'


@dataclass
class LoggerConfig:
    cfg_path: str
    save_dir: str = '.'


@dataclass
class ModelConfig:
    final_hid: int = 6


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    logger: LoggerConfig


@hydra.main(config_path='../config', config_name='base')
def main(cfg: Config):
    return cfg


def args_util():
    """
        Set the template of experiment parameters (in hydra.config_store)
    """
    cs = ConfigStore.instance()
    cs.store(group='trainer', name='base_train', node=TrainerConfig)
    cs.store(group='model', name='base_model', node=ModelConfig)
    cs.store(group='data', name='base_train', node=DataConfig)
    cs.store(group='logger', name='base_base', node=LoggerConfig)


if __name__ == '__main__':
    args_util()
