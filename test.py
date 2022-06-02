import torch
import hydra

from config import args_util, cfg2ep_crt
from logger import setup_logging
from model.model import Model
from tester.base import Tester
from data.datamodule import DataModule
from model.criterions import Loss


def cross_valid_test(cfg):
    datamodule = DataModule(cfg.data)
    test_loader = datamodule.test_dataloader()
    model = NewModel(cfg.model)
    tester = Tester(model=model, config=cfg.trainer, device=cfg.trainer.device,
                              data_loader=test_loader, criterion=Loss,
                              epoch_criterion=cfg2ep_crt.get(cfg.trainer.epoch_criterion, None))
    tester.test()


def predict(cfg):
    raise NotImplementedError

@hydra.main(config_path='../config', config_name='cross_valid_test')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name='test.log')
    torch.set_printoptions(precision=5)
    cross_valid_test(cfg=configs)


if __name__ == '__main__':
    args_util()
    main()
