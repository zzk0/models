import os
import time
import numpy as np
import tritonclient.http as httpclient
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer


def send_text(triton_client, text: str):
    text_bytes = []
    for sentence in text:
        text_bytes.append(str.encode(sentence, encoding='UTF-8'))
    text_np = np.array([text_bytes], dtype=np.object_)
    inputs = []
    inputs.append(httpclient.InferInput('pipeline_input', text_np.shape, "BYTES"))
    inputs[0].set_data_from_numpy(text_np, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('pipeline_output', binary_data=False))
    results = triton_client.infer('text_cnn_bert_pipeline', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('pipeline_output')
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


def send_ids(triton_client, input_ids):
    inputs = []
    inputs.append(httpclient.InferInput('input', input_ids.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_ids, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    results = triton_client.infer('text_cnn_bert_tensorrt', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')
    return output_data0

 
def accuracy():
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

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
        inputs_ids = np.array(inputs_ids).astype(np.int32)
        res = send_ids(triton_client, inputs_ids)
        print(items[i], res)
        res = 0 if res[0][0] > res[0][1] else 1
        correct_count += (res == tags[i])
    for i in tqdm(range(12500, 12510)):
        inputs = tokenizer.encode_batch([items[i]])
        inputs_ids = []
        for input in inputs:
            inputs_ids.append(input.ids)
        inputs_ids = np.array(inputs_ids).astype(np.int32)
        res = send_ids(triton_client, inputs_ids)
        print(items[i], res)
        res = 0 if res[0][0] > res[0][1] else 1
        correct_count += (res == tags[i])
    t1 = time.time()
    print('time cost: ', t1 - t0, correct_count / 20)


def send_text_to_preprocess(triton_client, text: str):
    text_bytes = []
    for sentence in text:
        text_bytes.append(str.encode(sentence, encoding='UTF-8'))
    text_np = np.array([text_bytes], dtype=np.object_)
    inputs = []
    inputs.append(httpclient.InferInput('preprocess_input', text_np.shape, "BYTES"))
    inputs[0].set_data_from_numpy(text_np, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('preprocess_output', binary_data=False))
    results = triton_client.infer('text_preprocess', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('preprocess_output')
    return output_data0


def preprocess(text: str):
    tokenizer = Tokenizer.from_file('./pretrained/bert-base-cased/tokenizer.json')
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)
    inputs = tokenizer.encode_batch(text)
    inputs_ids = []
    for input in inputs:
        inputs_ids.append(input.ids)
    inputs_ids = np.array(inputs_ids).astype(np.int32)
    return inputs_ids


def accuracy1():
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    items, tags = load_text()
    correct_count = 0
    t0 = time.time()
    for i in tqdm(range(10)):
        print(preprocess([items[i]]))
        res = send_text_to_preprocess(triton_client, [items[i]])
        print(res)
        res = send_ids(triton_client, res)
        print(items[i], res)
        res = 0 if res[0][0] > res[0][1] else 1
        correct_count += (res == tags[i])
    for i in tqdm(range(12500, 12510)):
        print(preprocess([items[i]]))
        res = send_text_to_preprocess(triton_client, [items[i]])
        print(res)
        res = send_ids(triton_client, res)
        print(items[i], res)
        res = 0 if res[0][0] > res[0][1] else 1
        correct_count += (res == tags[i])
    t1 = time.time()
    print('time cost: ', t1 - t0, correct_count / 20)


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    # items, tags = load_text()
    # correct_count = 0
    # t0 = time.time()
    # for i in tqdm(range(10)):
    #     res = send_text(triton_client, [items[i]])
    #     print(items[i], res)
    #     res = 0 if res[0][0] > res[0][1] else 1
    #     correct_count += (res == tags[i])
    # for i in tqdm(range(12500, 12510)):
    #     res = send_text(triton_client, [items[i]])
    #     print(items[i], res)
    #     res = 0 if res[0][0] > res[0][1] else 1
    #     correct_count += (res == tags[i])
    # t1 = time.time()
    # print('time cost: ', t1 - t0, correct_count / 20)


    accuracy1()
    # accuracy()
