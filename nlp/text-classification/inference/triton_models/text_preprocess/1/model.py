import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from tokenizers import Tokenizer


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(self.model_config, 'preprocess_output')
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.json')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128)

    def execute(self, requests):
        output_dtype = self.output_dtype
        responses = []
        for request in requests:
            input = pb_utils.get_input_tensor_by_name(request, 'preprocess_input')
            input = input.as_numpy()
            inputs = []
            for i in range(len(input)):
                inputs.append(str(input[i][0].decode('UTF-8')))
            input_ids = self.preprocess(inputs)
            out_tensor = pb_utils.Tensor('preprocess_output', input_ids.astype(output_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')

    def preprocess(self, text: str):
        inputs = self.tokenizer.encode_batch(text)
        inputs_ids = []
        for input in inputs:
            inputs_ids.append(input.ids)
        inputs_ids = np.array(inputs_ids)
        return inputs_ids
