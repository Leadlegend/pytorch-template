import torch
import hydra

from logger import setup_logging
from model.criterions import Loss
from trainer.base import Trainer
from data.datamodule import DataModule
from model.model import Model
from config import args_util, cfg2opt, cfg2sch, cfg2ep_crt


def init_optimizer(cfg, model):
    opt, lr, sched, epc = cfg.optimizer.lower(), cfg.lr, cfg.scheduler, cfg.epoch_criterion
    optimizer = cfg2opt[opt](params=model.parameters(), lr=lr)
    scheduler = cfg2sch.get(sched, None)
    epoch_criterion = cfg2ep_crt.get(epc, None)
    if scheduler is not None:
        scheduler = scheduler(optimizer=optimizer)
    return optimizer, scheduler, epoch_criterion


def train(cfg):
    datamodule = DataModule(cfg.data)
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    model = Model(cfg.model)
    optim, sched, epoch = init_optimizer(cfg.trainer, model)
    trainer = Trainer(model=model, config=cfg.trainer, device=cfg.trainer.device,
                              data_loader=train_loader, valid_data_loader=val_loader,
                              optimizer=optim, lr_scheduler=sched,
                              criterion=Loss, epoch_criterion=epoch)
    trainer.train()


@hydra.main(config_path='../config', config_name='base')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path)
    torch.set_printoptions(precision=5)
    train(cfg=configs)


if __name__ == '__main__':
    args_util()
    main()
