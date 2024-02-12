from collections.abc import Generator
from contextlib import nullcontext
from dataclasses import dataclass

import pandas as pd
import tiktoken
import torch
import torch.nn.functional as nn_func
from tiktoken import Encoding

from revllm.gpt import GPT


class TokenizerWrapper:
    def __init__(self, device_type: str = "cpu"):
        self.device = device_type
        self.dtype = torch.long
        self.tokenizer: Encoding = tiktoken.get_encoding("gpt2")

    def get_vocab_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    def get_all_tokens(self) -> list[str]:
        return [self.decode_single_token(token) for token in range(self.get_vocab_size())]

    def encode(self, text: str) -> torch.Tensor:
        x = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        x = torch.tensor(x, dtype=self.dtype, device=self.device)
        x = x.view(1, -1)
        return x

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def decode_single_token(self, token: int) -> str:
        return ascii(self.tokenizer.decode_single_token_bytes(token))[2:-1]

    def decode_tokens_separately(self, tokens: list[int]) -> list[str]:
        byte_tokens = self.tokenizer.decode_tokens_bytes(tokens)
        tokens = [ascii(token)[2:-1] for token in byte_tokens]
        return tokens


class PromptImportance:
    def __init__(
        self,
        prompt: str,
        input_token_ids: list[int] | torch.Tensor,
        input_tokens: list[str],
        input_token_scores: list[float] | torch.Tensor,
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


@dataclass
class LogitLensData:
    hidden_state_most_likely_token_df: pd.DataFrame
    hidden_state_most_likely_token_ids: torch.Tensor
    hidden_state_logits: torch.Tensor
    hidden_state_probabilities: torch.Tensor
    hidden_state_max_logits: torch.Tensor
    output_token_ids: int
    output_token: str


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
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 200,
        include_diagnostics: bool = False,
    ) -> str:
        """
        temperature: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k: retain only the top_k most likely tokens, clamp others to have 0 probability
        """
        x = self.tokenizer.encode(prompt)
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    include_diagnostics=include_diagnostics,
                )
                return self.tokenizer.decode(y[0].tolist())

    def run_logit_lens(self, prompt: str):
        x = self.tokenizer.encode(prompt)
        with torch.no_grad():
            with self.ctx:
                x = self.model.truncate_to_block_size(x)
                output = self.model.forward_with_diagnostics(x)

                hidden_state_logits = torch.cat(
                    output.hidden_state_logits, dim=0
                )  # num_layers, num_tokens, n_vocab
                device = hidden_state_logits.device
                num_layers, num_tokens, n_vocab = hidden_state_logits.size()
                hidden_state_most_likely_token_ids = torch.argmax(hidden_state_logits, dim=-1)
                pos = torch.arange(0, num_tokens, dtype=torch.long, device=device)
                hidden_state_max_logits = hidden_state_logits[
                    :, pos, hidden_state_most_likely_token_ids
                ][0]

        hidden_state_most_likely_token = [
            [
                self.tokenizer.decode_single_token(hidden_state_most_likely_token_ids[i, j])
                for j in range(num_tokens)
            ]
            for i in range(num_layers)
        ]
        output_token_id = torch.argmax(output.logits[0, 0, :]).item()
        output_token = self.tokenizer.decode_single_token(output_token_id)

        input_tokens = self.tokenizer.decode_tokens_separately(x[0].tolist())
        columns = [f"{i}_{t}" for i, t in enumerate(input_tokens)]
        hidden_state_most_likely_token_df = pd.DataFrame(
            hidden_state_most_likely_token, columns=columns
        )
        hidden_state_probabilities = nn_func.softmax(hidden_state_logits, dim=-1)
        output = LogitLensData(
            hidden_state_most_likely_token_df,
            hidden_state_most_likely_token_ids,
            hidden_state_logits,
            hidden_state_probabilities,
            hidden_state_max_logits,
            output_token_id,
            output_token,
        )
        return output

    def yield_importance_shap(self, prompt: str) -> Generator[PromptImportance, None, None]:
        pass

    def yield_importance_lime(self, prompt: str) -> Generator[PromptImportance, None, None]:
        pass

    def yield_importance_sequential_integrated_gradients(
        self, prompt: str, n_steps: int = 50
    ) -> Generator[PromptImportance, None, None]:
        input_token_ids = self.tokenizer.encode(prompt)
        while True:
            input_tokens = self.tokenizer.decode_tokens_separately(
                input_token_ids[0].detach().tolist()
            )  # [1, context_length]
            input_embeddings = self.model.transformer.wte(input_token_ids).to(
                self.device_type
            )  # [1, context_length, embedding_dimension]
            baseline_input_ids = torch.zeros_like(input_token_ids).to(
                self.device_type
            )  # [1, context_length]
            baseline_embeddings = self.model.transformer.wte(baseline_input_ids).to(
                self.device_type
            )  # [1, context_length, embedding_dimension]

            self.model.eval()
            output_logits = self.model(input_token_ids)[0]  # [1, 1, 50257]
            next_token_logits = output_logits[0, 0, :]  # [50257]
            predicted_token_id = torch.argmax(next_token_logits).item()  # [1]

            position_ids = (
                torch.arange(0, input_embeddings.size(1)).unsqueeze(0).to(self.device_type)
            )  # [1, context_length]
            position_embeddings = self.model.transformer.wpe(
                position_ids
            )  # [1, context_length, embedding_dimension]

            igs = torch.zeros_like(baseline_embeddings).to(
                self.device_type
            )  # [1, context_length, embedding_dimension]
            for target_word_index in range(input_embeddings.size(1)):
                target_word_embedding = input_embeddings[0, target_word_index, :].unsqueeze(
                    0
                )  # [1, embedding_dimension]
                target_word_baseline = baseline_embeddings[0, target_word_index, :].unsqueeze(
                    0
                )  # [1, embedding_dimension]

                alphas = (
                    torch.linspace(0, 1, steps=n_steps).unsqueeze(-1).to(self.device_type)
                )  # [n_steps, 1]

                step_embeddings = target_word_baseline + alphas * (
                    target_word_embedding - target_word_baseline
                )  # [n_steps, embedding_dimension]
                step_embeddings.requires_grad_(True)  # [n_steps, embedding_dimension]
                step_embeddings.retain_grad()
                step_embeddings.grad = None

                forward_embeddings = input_embeddings.repeat(
                    n_steps, 1, 1
                )  # [n_steps, context_length, embedding_dimension]
                forward_embeddings[:, target_word_index, :] = step_embeddings
                forward_embeddings = (
                    forward_embeddings + position_embeddings
                )  # [n_steps, context_length, embedding_dimension]
                forward_embeddings = self.model.transformer.drop(
                    forward_embeddings
                )  # [n_steps, context_length, embedding_dimension]

                for block in self.model.transformer.h:
                    forward_embeddings = block(
                        forward_embeddings
                    )  # [n_steps, context_length, embedding_dimension]

                forward_embeddings = self.model.transformer.ln_f(
                    forward_embeddings
                )  # [n_steps, context_length, embedding_dimension]
                output_at_step = self.model.lm_head(
                    forward_embeddings
                )  # [n_steps, context_length, 50257]

                class_output_at_step = output_at_step[:, -1, predicted_token_id]  # [n_steps]
                class_output_at_step.backward(
                    torch.ones_like(class_output_at_step),
                    retain_graph=True,
                )

                assert step_embeddings.grad is not None
                target_word_igs = step_embeddings.grad.mean(dim=0)  # [1, embedding_dimension]

                target_word_igs = target_word_igs * (
                    target_word_embedding - target_word_baseline
                )  # [1, embedding_dimension]
                igs[
                    :, target_word_index, :
                ] = target_word_igs  # [1, context_length, embedding_dimension]
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

    def yield_importance_integrated_gradients(
        self, prompt: str, n_steps: int = 50
    ) -> Generator[PromptImportance, None, None]:
        input_token_ids = self.tokenizer.encode(prompt)
        while True:
            input_tokens = self.tokenizer.decode_tokens_separately(
                input_token_ids[0].detach().tolist()
            )  # [1, context_length]
            input_embeddings = self.model.transformer.wte(input_token_ids).to(
                self.device_type
            )  # [1, context_length, embedding_dimension]
            embedding_dimension = input_embeddings.shape[-1]
            context_length = input_embeddings.shape[1]
            baseline_input_ids = torch.zeros_like(input_token_ids).to(
                self.device_type
            )  # [1, context_length]
            baseline_embeddings = self.model.transformer.wte(baseline_input_ids).to(
                self.device_type
            )  # [1, context_length, embedding_dimension]

            self.model.eval()
            output_logits = self.model(input_token_ids)[0]  # [1, 1, 50257]
            next_token_logits = output_logits[0, 0, :]  # [50257]
            predicted_token_id = torch.argmax(next_token_logits).item()  # [1]

            position_ids = (
                torch.arange(0, input_embeddings.size(1)).unsqueeze(0).to(self.device_type)
            )  # [1, context_length]
            position_embeddings = self.model.transformer.wpe(
                position_ids
            )  # [1, context_length, embedding_dimension]

            alphas = (
                torch.linspace(0, 1, steps=n_steps).unsqueeze(-1).unsqueeze(-1).to(self.device_type)
            )  # [n_steps, 1, 1]

            # expand all embeddings to n_steps
            zeros_to_expand_embeddings = torch.zeros(
                n_steps, context_length, embedding_dimension
            ).to(self.device_type)  # [n_steps, context_length, embedding_dimension]
            input_embeddings_expanded = (
                zeros_to_expand_embeddings + input_embeddings
            )  # [n_steps, context_length, embedding_dimension]
            baseline_embeddings_expanded = (
                zeros_to_expand_embeddings + baseline_embeddings
            )  # [n_steps, context_length, embedding_dimension]
            position_embeddings_expanded = (
                zeros_to_expand_embeddings + position_embeddings
            )  # [n_steps, context_length, embedding_dimension]

            input_embeddings_expanded_for_path = (
                input_embeddings_expanded.clone()
            )  # [n_steps, context_length, embedding_dimension]
            input_embeddings_at_steps = baseline_embeddings_expanded + alphas * (
                input_embeddings_expanded_for_path - baseline_embeddings_expanded
            )
            input_embeddings_at_steps.requires_grad_(True)
            input_embeddings_at_steps.retain_grad()

            forward_embeddings = (
                input_embeddings_at_steps + position_embeddings_expanded
            )  # [n_steps, context_length, embedding_dimension]
            forward_embeddings = self.model.transformer.drop(
                forward_embeddings
            )  # [n_steps, context_length, embedding_dimension]

            for block in self.model.transformer.h:
                forward_embeddings = block(
                    forward_embeddings
                )  # [n_steps, context_length, embedding_dimension]

            forward_embeddings = self.model.transformer.ln_f(
                forward_embeddings
            )  # [n_steps, context_length, embedding_dimension]
            output_logits_at_steps = self.model.lm_head(
                forward_embeddings
            )  # [n_steps, context_length, 50257]

            predicted_token_outputs_at_steps = output_logits_at_steps[
                :, -1, predicted_token_id
            ]  # [n_steps]

            predicted_token_outputs_at_steps.backward(
                torch.ones_like(predicted_token_outputs_at_steps), retain_graph=True
            )  # [n_steps]

            # extract gradients
            assert input_embeddings_at_steps.grad is not None
            ig_grads = (
                input_embeddings_at_steps.grad
            )  # [n_steps, context_length, embedding_dimension]
            ig_grads = ig_grads.mean(dim=0).unsqueeze(0)  # [1, context_length, embedding_dimension]

            igs = ig_grads * (
                input_embeddings - baseline_embeddings
            )  # [1, context_length, embedding_dimension]
            igs = igs.squeeze(0)

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
