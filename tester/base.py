import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm


class Tester:

    def __init__(self,
                 model,
                 criterion,
                 config,
                 device,
                 data_loader,
                 epoch_criterion=None):
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.epoch_criterion = epoch_criterion
        self.epoch_loss_name = '%s_correlation' % config.epoch_criterion if epoch_criterion is not None else ''

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.checkpoint = config.ckpt

        # setup visualization writer instance
        self.logger = logging.getLogger('Tester')
        self._resume_checkpoint(self.checkpoint)

    def test(self):
        result = self._test_epoch()
        # save logged informations into log dict
        log = dict()
        log.update(result)
        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('{:15s}: {}'.format(str(key), value))

    def _test_epoch(self):
        self.model.eval()
        log = dict()
        if self.epoch_criterion is not None:
            labels_pred, labels_gold = torch.zeros(0, 0).to(
                self.device), torch.zeros(0, 0).to(self.device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                data, target = batch.to(self.device)
                output = self.model(data)
                pred = self._out2pred(output)
                loss = self.criterion(pred, target)
                labels_pred, labels_gold = self._update_label(pred, target, labels_pred, labels_gold)
                #self.logger.info('%s\t%s' %(labels_pred.shape, labels_gold.shape))
                log.update({str(batch_idx): loss.item()})

            if self.epoch_criterion is not None:
                epoch_loss = self.epoch_criterion(labels_pred, labels_gold)
                log.update({self.epoch_loss_name: epoch_loss})
        return log

    def _out2pred(self, output):
        return output

    def _update_label(self, y_pred, y_gold):
        if self.epoch_criterion is not None:
            raise NotImplementedError

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        if not os.path.exists(resume_path):
            self.logger.error("Bad checkpoint path: {}".format(resume_path))
            sys.exit(1)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path,
                                map_location=torch.device(self.device))
        # we should make resume process robust to use various kinds of ckpt
        if 'state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['state_dict'],
                                           strict=False)
            except Exception as e:
                self.logger.error('Bad checkpoint format.', stack_info=True)
                raise ValueError
        else:
            # which means that we load the ckpt not to resume training, thus the params may not match perfectly.
            self.model.load_state_dict(checkpoint, strict=True)

        self.logger.info("Checkpoint loaded.")
