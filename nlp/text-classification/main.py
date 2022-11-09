import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf

from model import TextCNN
from dataset import TextClassificationDataModule, ImdbDataset


def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str, default='./cfg/text_cnn.yml')
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()


def get_model(cfg):
    if cfg.name == 'text_cnn':
        return TextCNN(cfg)
    return TextCNN(cfg)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    # build vocab
    imdb_dataset = ImdbDataset(cfg, 'train')
    cfg.vocab = imdb_dataset.vocab
    cfg.vocab_size = len(cfg.vocab)

    # data
    data_module = TextClassificationDataModule(cfg)

    # model
    model = get_model(cfg)

    # training
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.accelerator_devices,
        logger=pl.loggers.TensorBoardLogger(cfg.log_path, cfg.name),
        max_epochs=cfg.train.epochs
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
