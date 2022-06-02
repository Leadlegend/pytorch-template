import os
import sys
import torch
import logging
import subprocess
import numpy as np


class Trainer:

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 config,
                 device,
                 data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None):
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.start_epoch = 1
        self.save_period = config.save_period
        self.checkpoint_dir = config.save_dir
        self.epochs = config.epoch

        # setup visualization writer instance
        self.logger = logging.getLogger('Trainer')
        self.log_step = int(np.sqrt(len(data_loader)))
        self.logger.info('Logger Trainer will report info every %d steps' %
                         self.log_step)

        if not os.path.exists(self.checkpoint_dir):
            self.logger.warning(
                'Bad checkpoint save directory %s, try to create it.' %
                self.checkpoint_dir)
            try:
                subprocess.run('mkdir -p %s' % self.checkpoint_dir)
            except:
                raise ValueError('Cannot create directory %s' %
                                 self.checkpoint_dir)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.debug('{:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        log = dict()
        for batch_idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            log.update({str(batch_idx): loss.item()})

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        log = dict()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                log.update({str(batch_idx): loss.item()})
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch):
        model_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': model_dict,
            'optimizer': self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            state['scheduler'] = self.lr_scheduler.state_dict()
        filename = os.path.join(self.checkpoint_dir,
                                'ckpt-epoch{}.pt'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        if not os.path.exists(resume_path):
            self.logger.error("Bad checkpoint path: {}".format(resume_path))
            sys.exit(1)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path,
                                map_location=torch.device(self.device))
        # we should make resume process robust to use various kinds of ckpt
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if 'state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['state_dict'],
                                           strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as e:
                self.logger.error(
                    'Different model structture, optimizer or lr_scheduler, \
                    Please ensure you use the same configuration before resuming training.'                                                                                           ,
                    stack_info=True)
        else:
            # which means that we load the ckpt not to resume training, thus the params may not match perfectly.
            self.model.load_state_dict(checkpoint, strict=False)

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(
                self.start_epoch))
