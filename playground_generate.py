import torch

from revllm.helpers import reformat_lines
from revllm.model_wrapper import ModelWrapper

prompt = "Hello, my name is"
max_new_tokens = 50
temperature = 0.9

device = "gpu" if torch.cuda.is_available() else "cpu"
wrapper = ModelWrapper("gpt2", device_type=device, compiled=False)

generated_text = wrapper.generate(prompt, max_new_tokens, temperature, include_diagnostics=True)

print(generated_text)
generated_text = reformat_lines(generated_text, max_line_len=80)
