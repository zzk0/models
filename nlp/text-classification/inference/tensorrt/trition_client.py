import time
import numpy as np
import tritonclient.http as httpclient
from tokenizers import Tokenizer


def preprocess(text: str):
    tokenizer = Tokenizer.from_file('./pretrained/bert-base-uncased/tokenizer.json')
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)
    inputs = tokenizer.encode_batch(text)
    inputs_ids = []
    for input in inputs:
        inputs_ids.append(input.ids)
    inputs_ids = np.array(inputs_ids).astype(np.int32)
    return inputs_ids


def send_ids(triton_client, text: str):
    input_ids = preprocess(text)
    inputs = []
    inputs.append(httpclient.InferInput('input', input_ids.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_ids, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    results = triton_client.infer('text_cnn_bert_tensorrt', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')
    return output_data0


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


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    text = ["I went to the movie as a Sneak Preview in Austria. So didn't have an idea what I am going to see. The story is very normal. The movie is very long , I believe it could have cut to 1/2 without causing any problems to the story. Its the type of movie you can see in a boring night which you want to get bored more ! Ashton Kutcher was very good . Kevin Costner is OK. The movie is speaking about the US Coast Guards, how they are trained , their life style and the problems they face. As there aren't much effects in the movie. So if you want to watch it , then no need to waste your money and time going to the Cinema. Would be more effective to watch it at home when it gets on DVDs."]

    res = preprocess(text)
    print(res)

    res = send_text_to_preprocess(triton_client, text)
    print(res)

    # _ = send_ids(triton_client, text)
    # t0 = time.time()
    # res = send_ids(triton_client, text)
    # t1 = time.time()
    # print('send ids time cost: ', t1 - t0, res)

    # _ = send_text(triton_client, text)
    # t0 = time.time()
    # res = send_text(triton_client, text)
    # t1 = time.time()
    # print('send str time cost: ', t1 - t0, res)

