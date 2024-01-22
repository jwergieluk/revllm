import time
import warnings

import torch

from ig_demo_utils import *
warnings.filterwarnings("ignore")

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
start_time = time.time()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

context = "What is the capital of France?"

print(f"Initial context: {context}")
print("")
context_with_prediction = predict_hf(context)

print(context_with_prediction)
print("")

num_tokens_to_generate = 9

attributions_dict = {}
print("Predicted tokens, with attributions in order:")
print("(note: the first tokens are spaces)")

for _ in range(num_tokens_to_generate):
    predicted_token, predicted_token_id, input_ids, attention_mask, attributions = igs_nano(context)
    attributions_dict[predicted_token] = attributions

    # Append the predicted token ID to the input
    predicted_token_tensor = torch.tensor([[predicted_token_id]], dtype=torch.long).to(device)
    input_ids = torch.cat((input_ids, predicted_token_tensor), dim=1).to(device)
    context = tokenizer.decode(input_ids[0])

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")
print("")
print(tokenizer.decode(input_ids[0]))
