import os
import time
import numpy as np
import tritonclient.http as httpclient
from pathlib import Path
from tqdm import tqdm


def send_text(triton_client, text: str):
    text_bytes = []
    for sentence in text:
        text_bytes.append(str.encode(sentence, encoding='UTF-8'))
    text_np = np.array([text_bytes], dtype=np.object_)
    inputs = []
    inputs.append(httpclient.InferInput('input', text_np.shape, "BYTES"))
    inputs[0].set_data_from_numpy(text_np, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    results = triton_client.infer('text_cnn_bert_pipeline', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')
    return output_data0


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


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    items, tags = load_text()
    correct_count = 0
    t0 = time.time()
    for i in tqdm(range(len(items))):
        res = send_text(triton_client, [items[i]])
        res = 0 if res[0][0] > res[0][1] else 1
        correct_count += (res == tags[i])
    t1 = time.time()
    print('time cost: ', t1 - t0, correct_count / len(items))

