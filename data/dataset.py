import os
import torch

from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Optional, Union, List

from .tokenizer import Tokenizer


@dataclass
class Data:
    data
    label: Optional[float] = None


@dataclass
class Batch:
    data
    labels: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int):
        if idx:
            return self.labels
        else:
            return (self.cell_ids, self.drug_ids)

    def to(self, device):
        if self.labels is not None:
            return ((self.cell_ids.to(device), self.drug_ids.to(device)), self.labels.to(device))
        else:
            return ((self.cell_ids.to(device), self.drug_ids.to(device)), None)


class Dataset(Dataset):
    def __init__(self, cfg, tokenizer, sep='\t'):
        super().__init__()
        self.sep = sep
        self.data_map = list()
        self.lazy_mode: bool = cfg.lazy
        self.path: Union[str, List[str]] = cfg.path
        self.tokenizer: Tokenizer = tokenizer
        self.construct_dataset()

    def __getitem__(self, idx: int) -> Data:
        if self.lazy_mode:
            return self._lazy_get(idx)
        else:
            return self._get(idx)

    def __len__(self):
        return len(self.data_map)

    def _get(self, idx):
        return self.data_map[idx]

    def _lazy_get(self, idx):
        handler = self.data_map[idx]
        data = handler.readline().strip().split(self.sep)
        return self._parse_data(data)

    def construct_dataset(self):
        if not isinstance(self.path, str):
            for path in self.path:
                self._construct_dataset_file(str(path))
        else:
            self._construct_dataset_file(self.path)

    def _construct_dataset_file(self, path):
        if not os.path.exists(path):
            raise ValueError('Bad Dataset File: %s' % path)

        if not self.lazy_mode:
            with open(path, "r", encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    data = line.strip().split(self.sep)
                    self.data_map.append(self._parse_data(data))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    offset = f.tell() - len(line)
                    handler = open(path, 'r', encoding='utf-8')
                    handler.seek(offset)
                    self.data_map.append(handler)
        f.close()

    def _parse_data(self, data: tuple) -> Data:
        pass
