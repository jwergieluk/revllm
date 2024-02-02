import time

import torch

from revllm.model_wrapper import ModelWrapper

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wrapper = ModelWrapper("gpt2", device_type=device)

context = "What is the capital of France?"
num_tokens_to_generate = 9
print("Predicted tokens, with importance scores:")

#run whichever you want to see
# score_generator = model_wrapper.yield_importance_integrated_gradients(context)
score_generator = model_wrapper.yield_importance_sequential_integrated_gradients(context)

for _ in range(num_tokens_to_generate):
    score = next(score_generator)
    print(f"Predicted Token: {score.output_token}")
    print("")
    print(score.get_input_score_df())
    print("")

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")
