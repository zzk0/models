import os
import argparse
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tokenizers import Tokenizer

from model import TextCNN, TextRNN, TextRCNN, SpanEmo
from dataset import TextClassificationDataModule, ImdbDataset, SemEval18Dataset
from database import MLDatabase


def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str, default='./cfg/text_cnn_bert.yml')
    args.add_argument('-s', '--seed', type=int, default=42)
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()


def get_model(cfg):
    if cfg.name.startswith('text_cnn'):
        return TextCNN(cfg)
    elif cfg.name.startswith('text_rnn'):
        return TextRNN(cfg)
    elif cfg.name.startswith('text_rcnn'):
        return TextRCNN(cfg)
    elif cfg.name.startswith('span_emo'):
        return SpanEmo(cfg)
    else:
        raise NotImplemented


def write_sqlite(cfg, hypermeters, metrics):
    database = MLDatabase(cfg.sqlite_path)
    database.insert_experiment_result({
        'model': cfg.name,
        'dataset': cfg.dataset.name + cfg.dataset.get('lang', ''),
        'hypermeters': hypermeters,
        'metrics': metrics,
        'accuracy': metrics['accuracy'] if 'accuracy' in metrics else 0.0,
        'f1_micro': metrics['test_f1_micro'] if 'test_f1_micro' in metrics else 0.0,
        'f1_macro': metrics['test_f1_macro'] if 'test_f1_macro' in metrics else 0.0,
        'jaccard': metrics['test_jaccard'] if 'test_jaccard' in metrics else 0.0
    })
    database.close()


def export_onnx(cfg, model: pl.LightningModule):
    input_sample = [i for i in range(3000, 3000 + cfg.max_seq_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample, input_sample]).long()
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    model.to_onnx(os.path.join(cfg['onnx_path'], cfg.name + ".onnx"), input_sample, export_params=True, opset_version=11,
                  input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)


def test_pytorch_inference(cfg, model):
    input_sample = [i for i in range(3000, 3000 + cfg.max_seq_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample]).long()
    model = model.model
    model.eval()
    with torch.no_grad():
        y = model(input_sample)
        t0 = time.time()
        y = model(input_sample)
        t1 = time.time()
        print('time cost: ', t1 - t0, y)

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

 
def accuracy(model):
    model.eval()
    with torch.no_grad():
        tokenizer_path = os.path.join('./pretrained/bert-base-cased/tokenizer.json')
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=128)
        tokenizer.enable_padding(length=128)

        items, tags = load_text()
        correct_count = 0
        t0 = time.time()
        for i in tqdm(range(10)):
            inputs = tokenizer.encode_batch([items[i]])
            inputs_ids = []
            for input in inputs:
                inputs_ids.append(input.ids)
            inputs_ids = np.array(inputs_ids)
            inputs_ids = torch.Tensor(inputs_ids).int()
            print(inputs_ids)
            res = model(inputs_ids)
            print(items[i], res)
            res = 0 if res[0][0] > res[0][1] else 1
            correct_count += (res == tags[i])
        for i in tqdm(range(12500, 12510)):
            inputs = tokenizer.encode_batch([items[i]])
            inputs_ids = []
            for input in inputs:
                inputs_ids.append(input.ids)
            inputs_ids = np.array(inputs_ids)
            inputs_ids = torch.Tensor(inputs_ids).int()
            print(inputs_ids)
            res = model(inputs_ids)
            print(items[i], res)
            res = 0 if res[0][0] > res[0][1] else 1
            correct_count += (res == tags[i])
        t1 = time.time()
        print('time cost: ', t1 - t0, correct_count / 20)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    pl.seed_everything(args.seed, workers=True)
    if cfg.embedding.type not in ('bert'):
        # build vocab
        imdb_dataset = ImdbDataset(cfg, 'train')
        cfg.vocab = imdb_dataset.vocab
        cfg.vocab_size = len(cfg.vocab)
    if cfg.name.startswith('span_emo'):
        semeval18_dataset = SemEval18Dataset(cfg, 'train')
        cfg.dataset.size = len(semeval18_dataset)

    # data
    data_module = TextClassificationDataModule(cfg)

    # model
    model = get_model(cfg)

    # training
    if cfg.name.startswith('span_emo'):
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("./checkpoints", cfg.name), save_top_k=1, monitor="overall_val_loss")
        trainer = pl.Trainer(
            precision=cfg.train.precision,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.accelerator_devices,
            logger=pl.loggers.TensorBoardLogger(cfg.log_path, cfg.name),
            log_every_n_steps=cfg.log_every_n_steps,
            max_epochs=cfg.train.epochs,
            deterministic=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                checkpoint_callback
            ]
        )
        trainer.fit(model, data_module)
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, cfg=cfg)
        trainer.test(model, data_module)
    else:
        trainer = pl.Trainer(
            precision=cfg.train.precision,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.accelerator_devices,
            logger=pl.loggers.TensorBoardLogger(cfg.log_path, cfg.name),
            log_every_n_steps=cfg.log_every_n_steps,
            max_epochs=cfg.train.epochs,
            deterministic=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ]
        )
        model = model.load_from_checkpoint(checkpoint_path='/home/percent1/models/nlp/text-classification/logs/text_cnn_bert/version_1/checkpoints/epoch=9-step=980.ckpt', cfg=cfg)
        # trainer.fit(model, data_module)
        # trainer.test(model, data_module)
        if 'onnx_path' in cfg:
            accuracy(model)
            export_onnx(cfg, model)
            test_pytorch_inference(cfg, model)
    write_sqlite(cfg, model.hypermeters() + ', seed=' + str(args.seed), trainer.callback_metrics)




if __name__ == '__main__':
    main()
