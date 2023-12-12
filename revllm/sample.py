"""
Code based on Andrej Karpathy's nanoGPT model library

Sample from a trained model
"""
from contextlib import nullcontext
import torch
import tiktoken
from gpt import GPTConfig, GPT


class TokenizerWrapper:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def get_vocab_size(self) -> int:
        return 0

    def encode(self, text: str) -> torch.Tensor:
        x =  self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        x = x.view(1, -1)
        return x

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class ModelWrapper:
    def __init__(self, model_name: str):
        device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        compile = False # use PyTorch 2.0 to compile the model to be faster
        #exec(open('configurator.py').read()) # overrides from command line or config file

        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.model = GPT.from_pretrained(model_name, dict(dropout=0.0))
        self.model.eval()
        self.model.to(device)
        if compile:
            self.model: GPT = torch.compile(self.model) # requires PyTorch 2.0 (optional)
        self.tokenizer = TokenizerWrapper(device=device)


    def __str__(self):
        return str(self.model)


    def generate(self, prompt: str):
        x = self.tokenizer.encode(prompt)

        num_samples = 1  # number of samples to draw
        max_new_tokens = 150  # number of tokens generated in each sample
        temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        # run generation
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    return self.tokenizer.decode(y[0].tolist())


if __name__ == '__main__':
    model = ModelWrapper('gpt2')
    model.print()
    print(model.generate('Hello, my name is'))
