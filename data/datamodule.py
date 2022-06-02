import os
import logging

from functools import partial
from torch.utils.data import DataLoader

from .tokenizer import Tokenizer
from .dataset import Dataset
from .collate_fn import collate_fn


class DataModule:
    def __init__(self, cfg):
        self.logger = logging.getLogger('DataModule')
        self.cfg = cfg
        self.cell2idx = Tokenizer(cfg.cell2idx, has_index=True)
        self.drug2idx = Tokenizer(cfg.drug2idx, has_index=True)
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.setup()

    def setup(self):
        if self.cfg.train is not None:
            self.logger.info("Constructing Train Data...")
            self.train_dataset = Dataset(
                self.cfg.train, self.cell2idx, self.drug2idx)
        else:
            self.logger.warning('No Valid Train Data.')
        if self.cfg.val is not None:
            self.logger.info(" Constructing Validation Data...")
            self.val_dataset = Dataset(
                self.cfg.val, self.cell2idx, self.drug2idx)
        else:
            self.logger.warning('No Valid Val Data.')
        if self.cfg.test is not None:
            self.logger.info("Constructing Test Data...")
            self.test_dataset = Dataset(
                self.cfg.test, self.cell2idx, self.drug2idx)
        else:
            self.logger.warning('No Valid Test Data.')

    def train_dataloader(self):

        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=self.cfg.train.batch_size,
                                collate_fn=partial(
                                    collate_fn, labeled=self.cfg.train.label),
                                pin_memory=self.cfg.train.pin,
                                num_workers=self.cfg.train.workers,
                                shuffle=self.cfg.train.shuffle
                                )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(dataset=self.val_dataset,
                                batch_size=self.cfg.val.batch_size,
                                collate_fn=partial(
                                    collate_fn, labeled=self.cfg.val.label),
                                pin_memory=self.cfg.val.pin,
                                num_workers=self.cfg.val.workers,
                                shuffle=self.cfg.val.shuffle
                                )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(dataset=self.test_dataset,
                                batch_size=self.cfg.test.batch_size,
                                collate_fn=partial(
                                    collate_fn, labeled=self.cfg.test.label),
                                pin_memory=self.cfg.test.pin,
                                num_workers=self.cfg.test.workers,
                                shuffle=self.cfg.test.shuffle
                                )
        return dataloader
