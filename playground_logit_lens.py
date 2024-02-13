import torch

from revllm.model_wrapper import ModelWrapper

# Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters,
prompt = "The capital of Japan is the city of "

device = "gpu" if torch.cuda.is_available() else "cpu"
wrapper = ModelWrapper("gpt2", device_type=device, compiled=False)
logit_lens_data = wrapper.run_logit_lens(prompt)

all_tokens = wrapper.tokenizer.get_all_tokens()
weights = logit_lens_data.hidden_state_probabilities[6, -1].squeeze().numpy()

pass
