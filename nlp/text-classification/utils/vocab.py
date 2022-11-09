from tqdm import tqdm

UNK, PAD = '<UNK>', '<PAD>'

def build_vocab(items, tokenizer, max_size=10000, min_freq=2):
    vocab_dic = {}
    for line in tqdm(items):
        line = line.strip()
        if not line:
            continue
        for word in tokenizer(line):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(items, padding_size):
    map(lambda text: text.replace('<br />', ' '), items)
    tokenizer = lambda x: x.split(' ')
    vocab = build_vocab(items, tokenizer)
    input_ids = []
    for item in tqdm(items):
        item = tokenizer(item)
        if len(item) < padding_size:
            item.extend([PAD] * (padding_size - len(item)))
        else:
            item = item[:padding_size]
        sentence_id = []
        for word in item:
            sentence_id.append(vocab.get(word, vocab.get(UNK)))
        input_ids.append(sentence_id)
    return input_ids, vocab


def build_dev_and_test_dataset(items, padding_size, vocab):
    map(lambda text: text.replace('<br />', ' '), items)
    tokenizer = lambda x: x.split(' ')
    input_ids = []
    for item in tqdm(items):
        item = tokenizer(item)
        if len(item) < padding_size:
            item.extend([PAD] * (padding_size - len(item)))
        else:
            item = item[:padding_size]
        sentence_id = []
        for word in item:
            sentence_id.append(vocab.get(word, vocab.get(UNK)))
        input_ids.append(sentence_id)
    return input_ids, vocab
