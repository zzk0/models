import os
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Any, Dict


class ImdbSpanEmoDataset(Dataset):

    def __init__(self, cfg, dataset_type: str):
        super().__init__()
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.class_nums = 2
        self.max_length = self.cfg.max_seq_len
        self.segment_a = "positive negative"
        self.label_names = ["positive", "negative"]
        if self.cfg.embedding.use_local:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.name)

        self.items = []
        self.tags = []
        self.targets = []
        self.inputs = []
        self.labels = []
        self.input_lengths = []
        self.label_idxs = []

        self.load_text()
        self.tokenize_and_tensorize()

    def get_split_name(self, dataset_type):
        if dataset_type in ('dev', 'test'):
            return 'test'
        return 'train'

    def load_text(self):
        pos = os.path.join(self.cfg.dataset.path, self.get_split_name(self.dataset_type), 'pos')
        for file in tqdm(Path(pos).glob('*.txt'), 'loading pos dataset: '):
            self.items.append(file.read_text())
            self.tags.append(1)
            self.targets.append([0, 1])
        neg = os.path.join(self.cfg.dataset.path, self.get_split_name(self.dataset_type), 'neg')
        for file in tqdm(Path(neg).glob('*.txt'), 'loading neg dataset: '):
            self.items.append(file.read_text())
            self.tags.append(0)
            self.targets.append([1, 0])

    def preprocess(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def tokenize_and_tensorize(self):
        items = self.items
        map(self.preprocess, items)
        for input in tqdm(items):
            input_id, input_length, label_idxs = self.process_input(input, None)
            self.inputs.append(input_id)
            self.input_lengths.append(input_length)
            self.label_idxs.append(label_idxs)
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.labels = torch.Tensor(self.tags).type(torch.LongTensor)
        self.targets = torch.Tensor(self.targets).float()
        self.label_idxs = torch.tensor(self.label_idxs, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, index) -> Dict[str, Any]:
        return {
            'input_ids': self.inputs[index],
            'labels': self.labels[index],
            'targets': self.targets[index],
            'length': self.input_lengths[index],
            'label_idxs': self.label_idxs[index]
        }

    def process_input(self, data, labels):
        x = data
        x = self.tokenizer.encode_plus(self.segment_a,
                                        x,
                                        add_special_tokens=True,
                                        max_length=self.max_length,
                                        pad_to_max_length=True,
                                        truncation=True)
        input_id = x['input_ids']
        input_length = len([i for i in x['attention_mask'] if i == 1])
        label_idxs = [self.tokenizer.convert_ids_to_tokens(input_id).index(self.label_names[idx])
                        for idx, _ in enumerate(self.label_names)]
        return input_id, input_length, label_idxs


if __name__ == '__main__':
    # python3 dataset/imdb_spanemo_dataset.py
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('./cfg/span_emo_imdb.yml')
    dataset = ImdbSpanEmoDataset(cfg, 'train')
    print(dataset[10])
    print(dataset[101])
    print(dataset[101]['labels'])
    print(dataset[20000])
    print(len(dataset))
    print(dataset[10]['input_ids'].shape)
