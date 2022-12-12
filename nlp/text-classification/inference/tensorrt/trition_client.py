import time
import numpy as np
import tritonclient.http as httpclient
from tokenizers import Tokenizer


device_to_service = {
    'cpu': ['text_cnn_bert_openvino', 'text_cnn_bert_pipeline_cpu'],
    'gpu': ['text_cnn_bert_tensorrt', 'text_cnn_bert_pipeline'],
}

tokenizer = Tokenizer.from_file('/home/pretrained/bert-base-cased/tokenizer.json')
tokenizer.enable_truncation(max_length=128)
tokenizer.enable_padding(length=128)


def preprocess(text: str):
    inputs = tokenizer.encode_batch(text)
    inputs_ids = []
    for input in inputs:
        inputs_ids.append(input.ids)
    inputs_ids = np.array(inputs_ids).astype(np.int32)
    return inputs_ids


def send_ids(triton_client, text: str, device: str = 'gpu'):
    input_ids = preprocess(text)
    inputs = []
    inputs.append(httpclient.InferInput('input', input_ids.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_ids, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    results = triton_client.infer(device_to_service[device][0], inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')
    return output_data0


def send_text(triton_client, text: str, device: str = 'gpu'):
    text_bytes = []
    for sentence in text:
        text_bytes.append(str.encode(sentence, encoding='UTF-8'))
    text_np = np.array([text_bytes], dtype=np.object_)
    inputs = []
    inputs.append(httpclient.InferInput('input', text_np.shape, "BYTES"))
    inputs[0].set_data_from_numpy(text_np, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    results = triton_client.infer(device_to_service[device][1], inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')
    return output_data0



if __name__ == '__main__':
    device = 'gpu'

    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    text = ["I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge."]

    _ = send_ids(triton_client, text, device)
    t0 = time.time()
    res = send_ids(triton_client, text, device)
    t1 = time.time()
    print('send ids time cost: ', t1 - t0, res)

    _ = send_text(triton_client, text, device)
    t0 = time.time()
    res = send_text(triton_client, text, device)
    t1 = time.time()
    print('send str time cost: ', t1 - t0, res)

