import torch

from demo import get_model_wrapper, reformat_lines

prompt = "Hello, my name is"
max_new_tokens = 50
temperature = 0.9

device = "gpu" if torch.cuda.is_available() else "cpu"
wrapper = get_model_wrapper("gpt2", device_name=device)

generated_text = wrapper.generate(prompt, max_new_tokens, temperature)

print(generated_text)
generated_text = reformat_lines(generated_text, max_line_len=80)
