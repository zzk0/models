import os
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Any, Dict
from utils import build_dataset, build_dev_and_test_dataset


class ImdbDataset(Dataset):

    def __init__(self, cfg, dataset_type: str):
        super().__init__()
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.label_map = {'pos': 1, 'neg': 0}
        self.items = []
        self.tags = []
        self.inputs = None
        self.labels = None
        self.vocab = {}
        self.load_text()
        if self.cfg.embedding.type in ('bert'):
            self.tokenize_and_tensorize()
        else:
            if dataset_type in ('train'):
                self.inputs, self.vocab = build_dataset(self.items, self.cfg.max_seq_len)
            else:
                self.inputs, _ =  build_dev_and_test_dataset(self.items, self.cfg.max_seq_len, dict(self.cfg.vocab))
            self.labels = self.tags
            self.inputs = {'input_ids': torch.Tensor(self.inputs).type(torch.LongTensor)}
            self.labels = torch.Tensor(self.labels).type(torch.LongTensor)

    def get_split_name(self, dataset_type):
        if dataset_type in ('dev', 'test'):
            return 'test'
        return 'train'

    def load_text(self):
        pos = os.path.join(self.cfg.dataset.path, self.get_split_name(self.dataset_type), 'pos')
        for file in tqdm(Path(pos).glob('*.txt'), 'loading pos dataset: '):
            self.items.append(file.read_text())
            self.tags.append(self.label_map['pos'])
        neg = os.path.join(self.cfg.dataset.path, self.get_split_name(self.dataset_type), 'neg')
        for file in tqdm(Path(neg).glob('*.txt'), 'loading neg dataset: '):
            self.items.append(file.read_text())
            self.tags.append(self.label_map['neg'])

    def preprocess(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def tokenize_and_tensorize(self):
        if self.cfg.embedding.type == 'bert':
            self.inputs = self.items
            map(self.preprocess, self.inputs)
            if self.cfg.embedding.use_local:
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.path, local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.name)
            # items contain three part: input_ids, token_type_ids, attentioin_mask
            self.inputs = tokenizer(self.inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.cfg.max_seq_len)
            self.labels = torch.Tensor(self.tags).type(torch.LongTensor)

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, index) -> Dict[str, Any]:
        return {
            # 'items': self.items[index],
            # 'tags': self.tags[index],
            'input_ids': self.inputs['input_ids'][index],
            'labels': self.labels[index]
        }


if __name__ == '__main__':
    # python3 dataset/imdb_dataset.py
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./cfg/text_cnn.yml')
    cfg.x = {1: 2, 2: 3}
    print(type(cfg.x))
    print(type(dict(cfg.x)))
    dataset = ImdbDataset(cfg, 'train')
    print(dataset[10])
    print(dataset[101])
    print(dataset[101]['labels'])
    print(dataset[20000])
    print(len(dataset))
    print(dataset[10]['input_ids'].shape)
