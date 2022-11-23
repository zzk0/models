import torch
import numpy as np
import argparse
import OpenAttack as oa
import datasets # use the Hugging Face's datasets library

from dataset import SemEval18Dataset
from model import SpanEmo
from omegaconf import OmegaConf


class MyClassifier(oa.Classifier):

    def __init__(self, cfg):
        self.victim = SpanEmo(cfg)
        model_path = '/home/percent1/models/nlp/text-classification/checkpoints/span_emo_english/epoch=3-step=428.ckpt'
        self.victim = self.victim.load_from_checkpoint(checkpoint_path=model_path, cfg=cfg)
        self.dataset = SemEval18Dataset(cfg, 'train')

    def __victim_forward(self, input_):
        batch = {
            'input_ids': [],
            'length': [],
            'label_idxs': []
        }
        for text in input_:
            input_id, input_length, label_idxs = self.dataset.process_input(text, None)
            batch['input_ids'].append(input_id)
            batch['length'].append(input_length)
            batch['label_idxs'].append(label_idxs)
        batch['input_ids'] = torch.Tensor(batch['input_ids']).long()
        batch['length'] = torch.Tensor(batch['length']).long()
        batch['label_idxs'] = torch.Tensor(batch['label_idxs']).long()
        _, _, logits, y_pred, _ = self.victim(batch)
        return logits, y_pred

    def get_pred(self, input_):
        _, y_pred = self.__victim_forward(input_)
        return y_pred

    def get_prob(self, input_):
        y_logits, _ = self.__victim_forward(input_)
        return y_logits


def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str, default='./cfg/span_emo_english.yml')
    args.add_argument('-s', '--seed', type=int, default=42)
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    
    victim = MyClassifier(cfg)
    dataset = SemEval18Dataset(cfg, 'train')
    dataset = datasets.Dataset.from_dict({
        'x': dataset.data,
        'y': dataset.labels
    })
    attacker = oa.attackers.BAEMCLttacker('/home/percent1/models/nlp/text-classification/pretrained/bert-base-uncased')
    metrics = []
    attack_eval = oa.AttackEval(attacker, victim, metrics=metrics)
    attack_eval.eval(dataset, visualize=True)

if __name__ == '__main__':
    main()
