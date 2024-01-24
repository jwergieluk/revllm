import json
from collections.abc import Generator
from contextlib import nullcontext

import pandas as pd
import tiktoken
import torch

from revllm.gpt import GPT


class TokenizerWrapper:
    def __init__(self, device_type: str = "cpu"):
        self.device = device_type
        self.dtype = torch.long
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def get_vocab_size(self) -> int:
        return 0

    def encode(self, text: str) -> torch.Tensor:
        x = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        x = torch.tensor(x, dtype=self.dtype, device=self.device)
        x = x.view(1, -1)
        return x

    def decode(self, tokens: list[int]) -> str:
        return json.dumps(self.tokenizer.decode(tokens))

    def decode_single_token(self, token: int) -> str:
        return json.dumps(
            self.tokenizer.decode_single_token_bytes(token).decode("utf-8", errors="ignore")
        )

    def decode_tokens_separately(self, tokens: list[int]) -> list[str]:
        text, offsets = self.tokenizer.decode_with_offsets(tokens)
        tokens = [
            text[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=False)
        ] + [
            text[offsets[-1] :],
        ]
        return [json.dumps(token) for token in tokens]


class PromptImportance:
    def __init__(
        self,
        prompt: str,
        input_token_ids: list[int],
        input_tokens: list[str],
        input_token_scores: list[float],
        output_token_id: int,
        output_token: str,
    ):
        self.prompt = prompt
        self.input_token_ids = (
            input_token_ids.squeeze().tolist()
            if isinstance(input_token_ids, torch.Tensor)
            else input_token_ids
        )
        self.input_tokens = input_tokens
        self.input_token_scores = (
            input_token_scores.squeeze().tolist()
            if isinstance(input_token_scores, torch.Tensor)
            else input_token_scores
        )
        self.output_token_id = (
            output_token_id.squeeze().item()
            if isinstance(output_token_id, torch.Tensor)
            else output_token_id
        )
        self.output_token = output_token

    def get_input_score_df(self):
        return pd.DataFrame(
            {
                "token": self.input_tokens,
                "token_id": self.input_token_ids,
                "score": self.input_token_scores,
            }
        )


class ModelWrapper:
    def __init__(
        self, model_name: str, device_type: str | torch.device = "cpu", compiled: bool = False
    ):
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        # exec(open('configurator.py').read()) # overrides from command line or config file

        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        if isinstance(device_type, torch.device):
            self.device_type = device_type.type
            self.device = device_type
        elif isinstance(device_type, str):
            self.device_type = device_type.lower()
            self.device = torch.device(self.device_type)
        else:
            raise ValueError(f"Invalid device: {device_type}")
        pt_dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=pt_dtype_map[dtype])
        )

        self.model_name = model_name
        self.model = GPT.from_pretrained(model_name, dict(dropout=0.0))
        self.model.eval()
        self.model.to(self.device)
        if compiled:
            self.model: GPT = torch.compile(self.model)  # requires PyTorch 2.0 (optional)
        self.tokenizer = TokenizerWrapper(device_type=device_type)

    def __str__(self):
        return str(self.model)

    def generate(
        self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 200
    ) -> str:
        """
        temperature: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k: retain only the top_k most likely tokens, clamp others to have 0 probability
        """
        x = self.tokenizer.encode(prompt)
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                return self.tokenizer.decode(y[0].tolist())

    def yield_importance_integrated_gradients(
        self, prompt: str, max_new_tokens: int = 10, n_steps: int = 50
    ) -> Generator[PromptImportance, None, None]:
        input_token_ids = self.tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            input_tokens = self.tokenizer.decode_tokens_separately(
                input_token_ids[0].detach().tolist()
            )  # [1, 7]
            input_embeddings = self.model.transformer.wte(input_token_ids).to(
                self.device_type
            )  # [1, 7, 768]
            baseline_input_ids = torch.zeros_like(input_token_ids).to(self.device_type)  # [1, 7]
            baseline_embeddings = self.model.transformer.wte(baseline_input_ids).to(
                self.device_type
            )  # [1, 7, 768]

            self.model.eval()
            output_logits = self.model(input_token_ids)[0]  # [1, 1, 50257]
            next_token_logits = output_logits[0, 0, :]  # [50257]
            predicted_token_id = torch.argmax(next_token_logits).item()  # [1]

            position_ids = (
                torch.arange(0, input_embeddings.size(1)).unsqueeze(0).to(self.device_type)
            )  # [1, 7]
            position_embeddings = self.model.transformer.wpe(position_ids)  # [1, 7, 768]

            igs = torch.zeros_like(baseline_embeddings).to(self.device_type)  # [1, 7, 768]
            for target_word_index in range(input_embeddings.size(1)):
                target_word_embedding = input_embeddings[0, target_word_index, :].unsqueeze(
                    0
                )  # [1, 768]
                target_word_baseline = baseline_embeddings[0, target_word_index, :].unsqueeze(
                    0
                )  # [1, 768]

                alphas = (
                    torch.linspace(0, 1, steps=n_steps).unsqueeze(-1).to(self.device_type)
                )  # [50, 1]

                step_embeddings = target_word_baseline + alphas * (
                    target_word_embedding - target_word_baseline
                )  # [50, 768]
                step_embeddings.requires_grad_(True)  # [50, 768]
                step_embeddings.retain_grad()
                step_embeddings.grad = None

                forward_embeddings = input_embeddings.repeat(n_steps, 1, 1)  # [50, 7, 768]
                forward_embeddings[:, target_word_index, :] = step_embeddings
                forward_embeddings = forward_embeddings + position_embeddings  # [50, 7, 768]
                forward_embeddings = self.model.transformer.drop(forward_embeddings)  # [50, 7, 768]

                for block in self.model.transformer.h:
                    forward_embeddings = block(forward_embeddings)  # [50, 7, 768]

                forward_embeddings = self.model.transformer.ln_f(forward_embeddings)  # [50, 7, 768]
                output_at_step = self.model.lm_head(forward_embeddings)  # [50, 7, 50257]

                class_output_at_step = output_at_step[:, -1, predicted_token_id]  # [50]
                summed_output_for_gradient_computation = class_output_at_step.sum()  # [1]
                summed_output_for_gradient_computation.backward(retain_graph=True)

                assert step_embeddings.grad is not None
                step_embeddings_grad_pre_sum = step_embeddings.grad / n_steps  # [50, 768]

                target_word_igs = step_embeddings_grad_pre_sum.sum(dim=0)  # [1, 768]
                target_word_igs = target_word_igs * (
                    target_word_embedding - target_word_baseline
                )  # [1, 768]
                igs[:, target_word_index, :] = target_word_igs  # [1, 7, 768]
            attributions = igs.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)

            predicted_token = self.tokenizer.decode_single_token(predicted_token_id)
            importance = PromptImportance(
                prompt,
                input_token_ids,
                input_tokens,
                attributions.tolist(),
                predicted_token_id,
                predicted_token,
            )
            yield importance
            input_token_ids = torch.cat(
                [
                    input_token_ids,
                    torch.tensor(
                        [[predicted_token_id]], dtype=self.tokenizer.dtype, device=self.device_type
                    ),
                ],
                dim=1,
            )


if __name__ == "__main__":
    model = ModelWrapper("gpt2")
    print(model)
    print(model.generate("Hello, my name is"))
