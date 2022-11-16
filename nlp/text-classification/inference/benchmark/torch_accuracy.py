import os
import time
import torch
import numpy as np
from model import TextCNN
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from tokenizers import Tokenizer


def load_text():
    label_map = {'pos': 1, 'neg': 0}
    items = []
    tags = []
    pos = os.path.join('./data/aclImdb/test/pos')
    for file in tqdm(Path(pos).glob('*.txt'), 'loading pos dataset: '):
        items.append(file.read_text())
        tags.append(label_map['pos'])
    neg = os.path.join('./data/aclImdb/test/neg')
    for file in tqdm(Path(neg).glob('*.txt'), 'loading neg dataset: '):
        items.append(file.read_text())
        tags.append(label_map['neg'])
    for i in range(len(items)):
        items[i] = items[i].replace('<br />', ' ')
    return items, tags


def preprocess(tokenizer, text: str):
    inputs = tokenizer.encode_batch(text)
    inputs_ids = []
    for input in inputs:
        inputs_ids.append(input.ids)
    inputs_ids = np.array(inputs_ids)
    return inputs_ids


if __name__ == '__main__':
    device = 'cpu' # 'cuda:0'
    cfg = OmegaConf.load('./cfg/text_cnn_bert.yml')
    model = TextCNN(cfg)
    model = model.load_from_checkpoint(checkpoint_path='logs/text_cnn_bert/version_1/checkpoints/epoch=9-step=980.ckpt', cfg=cfg)
    model.to(device)

    tokenizer_path = os.path.join('./pretrained/bert-base-cased/tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)

    items, tags = load_text()
    correct_count = 0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(items))):
            input_ids = preprocess(tokenizer, [items[i]])
            input_ids = torch.Tensor(input_ids).int().to(device)
            res = model(input_ids)
            res = 0 if res[0][0] > res[0][1] else 1
            correct_count += (res == tags[i])
    t1 = time.time()
    print('time cost: ', t1 - t0, correct_count / len(items))
