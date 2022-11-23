import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class SemEval18Dataset(Dataset):

    def __init__(self, cfg, dataset_type: str):
        super().__init__()
        self.cfg = cfg
        self.max_length = self.cfg.max_seq_len
        gold = '-gold' if dataset_type == 'test' else ''
        if self.cfg.dataset.lang == 'english':
            self.filename = os.path.join(self.cfg.dataset.path, "2018-E-c-En-{}{}.txt".format(dataset_type, gold))
            # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.path, do_lower_case=True)
            self.segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
            self.label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                                "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
        elif self.cfg.dataset.lang == 'arabic':
            self.filename = os.path.join(self.cfg.dataset.path, "2018-E-c-Ar-{}{}.txt".format(dataset_type, gold))
            self.segment_a = "غضب توقع قرف خوف سعادة حب تفأول اليأس حزن اندهاش أو ثقة؟"
            self.label_names = ['غضب', 'توقع', 'قر', 'خوف', 'سعادة', 'حب', 'تف', 'الياس', 'حزن', 'اند', 'ثقة']
        elif self.cfg.dataset.lang == 'spanish':
            self.filename = os.path.join(self.cfg.dataset.path, "2018-E-c-Es-{}{}.txt".format(dataset_type, gold))
            self.segment_a = "ira anticipación asco miedo alegría amor optimismo pesimismo tristeza sorpresa or confianza?"
            self.label_names = ['ira', 'anticip', 'asco', 'miedo', 'alegr', 'amor', 'optimismo',
                                'pesim', 'tristeza', 'sorpresa', 'confianza']

        do_lower_case = (self.cfg.dataset.lang == 'english')
        if self.cfg.embedding.use_local:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.path, do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.name, do_lower_case=do_lower_case)

        self.preprocessor = twitter_preprocessor()

        self.data, self.labels = self.load_dataset()
        self.inputs, self.lengths, self.label_indices = self.process_data()
        self.labels = torch.Tensor(self.labels).float()
        self.label_nums = 11

    def load_dataset(self):
        df = pd.read_csv(self.filename, sep='\t')
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        return x_train, y_train

    def load_text(self):
        pos = os.path.join(self.cfg.dataset.path, self.get_split_name(self.dataset_type), 'pos')
        for file in tqdm(Path(pos).glob('*.txt'), 'loading pos dataset: '):
            self.items.append(file.read_text())
            self.tags.append(self.label_map['pos'])
        neg = os.path.join(self.cfg.dataset.path, self.get_split_name(self.dataset_type), 'neg')
        for file in tqdm(Path(neg).glob('*.txt'), 'loading neg dataset: '):
            self.items.append(file.read_text())
            self.tags.append(self.label_map['neg'])

    def process_data(self):
        inputs, lengths, label_indices = [], [], []
        for x in tqdm(self.data, desc='PreProcessing dataset ..'):
            input_id, input_length, label_idxs = self.process_input(x, None)
            inputs.append(input_id)
            lengths.append(input_length)
            label_indices.append(label_idxs)            
        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return {
            'input_ids': inputs,
            'targets': labels,
            'length': length,
            'label_idxs': label_idxs
        }

    def __len__(self):
        return len(self.inputs)
    
    def process_input(self, data, labels):
        x = data
        x = ' '.join(self.preprocessor(x))
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
    # python3 dataset/semeval18_dataset.py
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./cfg/span_emo_english.yml')
    dataset = SemEval18Dataset(cfg, 'train')
    print(dataset[10])
    print(dataset[101])
    print(len(dataset))
    print(dataset.process_input("I'm the wholesome drunk that sends people memes and compliments them at 2am on snap", None))

