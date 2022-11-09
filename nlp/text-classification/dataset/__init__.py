import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig

from .imdb_dataset import ImdbDataset

class TextClassificationDataModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super(TextClassificationDataModule, self).__init__()
        self.cfg = cfg
        
    def get_dataset_class(self):
        if self.cfg.dataset.name == 'imdb':
            return ImdbDataset
        raise NotImplemented

    def setup(self, stage: str):
        dataset_class = self.get_dataset_class()
        if stage == 'fit':
            self.train_set = dataset_class(self.cfg, 'train')
            self.dev_set = dataset_class(self.cfg, 'dev')
        if stage == 'test':
            self.test_set = dataset_class(self.cfg, 'test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.cfg.train.train_batch_size, shuffle=True, num_workers=self.cfg.train.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dev_set, self.cfg.train.dev_batch_size, shuffle=True, num_workers=self.cfg.train.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.cfg.train.test_batch_size, shuffle=False, num_workers=self.cfg.train.num_workers, persistent_workers=True)

