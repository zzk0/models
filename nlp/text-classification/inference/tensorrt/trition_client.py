import time
import numpy as np
import tritonclient.http as httpclient
from tokenizers import Tokenizer


def preprocess(text: str):
    tokenizer = Tokenizer.from_file('./pretrained/bert-base-uncased/tokenizer.json')
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)
    input = tokenizer.encode(text)
    print(input.ids, input.tokens)
    print(len(input.ids))
    return input.ids


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    input_ids = preprocess('''
        I went and saw this movie last night after being coaxed to by a few friends of mine. 
        I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. 
        I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. 
        The sign of a good movie is that it can toy with our emotions. This one did exactly that. 
        The entire theater (which was sold out) was overcome by laughter during the first half of the movie, 
        and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, 
        but many full grown men as well, trying desperately not to let anyone see them crying. 
        This movie was great, and I suggest that you go see it before you judge.
    ''')

    N = 1
    # input_sample = [i for i in range(3000, 3000 + 128 - 2)]
    # input_sample = [101, *input_sample, 102]
    input_sample = np.array([input_ids for i in range(N)]).astype(np.int32)
    print(input_sample.shape)

    t0 = time.time()
    inputs = []
    inputs.append(httpclient.InferInput('input', input_sample.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_sample, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    results = triton_client.infer('text_cnn_bert_tensorrt', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')

    t1 = time.time()
    print(output_data0.shape)
    print('time cost: ', t1 - t0 ,output_data0)
