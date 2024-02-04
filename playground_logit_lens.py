import torch

from revllm.model_wrapper import ModelWrapper

# Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters,
prompt = "Specifically, we train GPT-3, an"

device = "gpu" if torch.cuda.is_available() else "cpu"
wrapper = ModelWrapper("gpt2", device_type=device, compiled=False)
logit_lens_data = wrapper.run_logit_lens(prompt)

print(logit_lens_data.hidden_state_most_likely_token_df)
