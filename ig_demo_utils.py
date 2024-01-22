import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from revllm.gpt import GPT

model_type = "gpt2"
model_hf = GPT2LMHeadModel.from_pretrained(model_type)
model_nano = GPT.from_pretrained(model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_hf.to(device)
model_nano.to(device)

tokenizer = GPT2Tokenizer.from_pretrained(model_type)

import torch


def predict_hf(context: str) -> str:
    encoded_input = tokenizer(context, return_tensors="pt").to(device)
    input_ids = encoded_input["input_ids"].to(device) # torch.Size([1, 7])
    attention_mask = encoded_input["attention_mask"].to(device) # torch.Size([1, 7])

    output = model_hf.generate(**encoded_input, max_length=len(input_ids) + 15, do_sample=False) # torch.Size([1, 22])

    return tokenizer.decode(output[0])


def igs_hf(context: str, n_steps: int = 50) -> tuple[str, int, torch.Tensor]:
    print("===========================================")

    encoded_input = tokenizer(context, return_tensors="pt") 
    input_ids = encoded_input["input_ids"].to(device)   # torch.Size([1, 7])
    attention_mask = encoded_input["attention_mask"].to(device) # torch.Size([1, 7])
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach()) 

    model_hf.eval()
    # with torch.no_grad():.....(not sure if I need this)
    output_logits = model_hf(input_ids=input_ids, attention_mask=attention_mask).logits.to(device) # torch.Size([1, 7, 50257])

    next_token_logits = output_logits[0, -1, :].to(device)  # torch.Size([50257])
    predicted_token_id = torch.argmax(next_token_logits).item() # int

    baseline_input_ids = torch.zeros_like(input_ids).to(device) # torch.Size([1, 7])
    baseline_embeddings = model_hf.transformer.wte(baseline_input_ids).to(device) # torch.Size([1, 7, 768])

    igs = torch.zeros_like(baseline_embeddings)
    for target_word_index in range(igs.size(1)):
        target_word_baseline = baseline_embeddings[0, target_word_index, :].to(device) # torch.Size([768])

        input_embeddings = model_hf.transformer.wte(input_ids).to(device) # torch.Size([1, 7, 768])
        target_word_embedding = input_embeddings[0, target_word_index, :].to(device) # torch.Size([768])

        position_ids = torch.arange(0, input_embeddings.size(1)).unsqueeze(0).to(device)    # torch.Size([1, 7])
        position_embeddings = model_hf.transformer.wpe(position_ids).to(device) # torch.Size([1, 7, 768])

        ig = torch.zeros_like(target_word_embedding)
        for step in range(n_steps):
            alpha = step / n_steps

            modified_embedding = target_word_baseline + alpha * (
                target_word_embedding - target_word_baseline
            ) # torch.Size([768])
            modified_embedding.requires_grad_(True) # torch.Size([768])
            modified_embedding.retain_grad() # torch.Size([768])

            if modified_embedding.grad is not None:
                modified_embedding.grad.zero_() # torch.Size([768])

            step_embeddings = input_embeddings.clone()  # torch.Size([1, 7, 768])
            step_embeddings.requires_grad_(True)
            step_embeddings[0, target_word_index, :] = modified_embedding # torch.Size([1, 7, 768])
            embeddings = step_embeddings + position_embeddings # torch.Size([1, 7, 768])
            embeddings = model_hf.transformer.drop(embeddings) # torch.Size([1, 7, 768])

            for block in model_hf.transformer.h:
                embeddings = block(embeddings)[0] # torch.Size([1, 7, 768])

            embeddings = model_hf.transformer.ln_f(embeddings) # torch.Size([1, 7, 768])
            output_at_step = model_hf.lm_head(embeddings) # torch.Size([1, 7, 50257])

            class_output_at_step = output_at_step[0, -1, predicted_token_id] # torch.Size([1, 7, 50257])
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as prof:

            class_output_at_step.backward(retain_graph=True) 

            assert modified_embedding.grad is not None

            ig_at_step = modified_embedding.grad / n_steps # torch.Size([768])
            ig += ig_at_step # torch.Size([768])

        ig *= target_word_embedding - target_word_baseline # torch.Size([768])
        igs[0, target_word_index, :] = ig # torch.Size([1, 7, 768])

    predicted_token = tokenizer.decode(predicted_token_id)

    attributions = igs.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    print(f"Predicted Token: {predicted_token}")
    print("")
    for key, value in sorted(zip(all_tokens, attributions, strict=False), key=lambda item: item[1], reverse=True):
        print(f"{key}: {value}")
    print("")

    return predicted_token, predicted_token_id, input_ids, attention_mask, attributions

def igs_nano(context: str, n_steps: int = 50) -> tuple[str, int, torch.Tensor]:

    print("===========================================")

    encoded_input = tokenizer(context, return_tensors="pt") #returns a dict
    input_ids = encoded_input["input_ids"].to(device)  # [1, 7]
    baseline_input_ids = torch.zeros_like(input_ids).to(device) #[1, 7]

    attention_mask = encoded_input["attention_mask"].to(device) # [1, 7]
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())    # [1, 7]

    input_embeddings = model_nano.transformer.wte(input_ids).to(device) #[1, 7, 768]
    baseline_embeddings = model_nano.transformer.wte(baseline_input_ids).to(device) #[1, 7, 768]

    model_nano.eval()

    output_logits = model_nano(input_ids)[0] #[1, 7, 50257]
    next_token_logits = output_logits[0, 0, :]  # [50257]
    predicted_token_id = torch.argmax(next_token_logits).item() # [1]

    position_ids = torch.arange(0, input_embeddings.size(1)).unsqueeze(0).to(device) #[1, 7]
    position_embeddings = model_nano.transformer.wpe(position_ids) #[1, 7, 768]

    igs = torch.zeros_like(baseline_embeddings).to(device) #[1, 7, 768]

    for target_word_index in range(input_embeddings.size(1)):
    
        target_word_embedding = input_embeddings[0,target_word_index,:].unsqueeze(0) #[1, 768]
        target_word_baseline = baseline_embeddings[0,target_word_index,:].unsqueeze(0) #[1, 768]

        alphas = torch.linspace(0, 1, steps=n_steps).unsqueeze(-1).to(device) #[50, 1]

        step_embeddings = target_word_baseline + alphas * (target_word_embedding - target_word_baseline) #[50, 768]
        step_embeddings.requires_grad_(True) #[50, 768]
        step_embeddings.retain_grad()
        step_embeddings.grad = None

        forward_embeddings = input_embeddings.repeat(n_steps, 1, 1) #[50, 7, 768]
        forward_embeddings[:,target_word_index,:] = step_embeddings
        forward_embeddings = forward_embeddings + position_embeddings #[50, 7, 768]
        forward_embeddings = model_nano.transformer.drop(forward_embeddings) #[50, 7, 768]   

        for block in model_nano.transformer.h:
            forward_embeddings = block(forward_embeddings) #[50, 7, 768]

        forward_embeddings = model_nano.transformer.ln_f(forward_embeddings) #[50, 7, 768]
        output_at_step = model_nano.lm_head(forward_embeddings) #[50, 7, 50257]

        class_output_at_step = output_at_step[:, -1, predicted_token_id] #[50]
        summed_output_for_gradient_computation = class_output_at_step.sum() #[1]
        summed_output_for_gradient_computation.backward(retain_graph=True)
        # class_output_at_step.backward(retain_graph=True)

        assert step_embeddings.grad is not None

        step_embeddings_grad_pre_sum = step_embeddings.grad/n_steps #[50, 768]

        target_word_igs = step_embeddings_grad_pre_sum.sum(dim=0) #[1, 768]
        target_word_igs = target_word_igs * (target_word_embedding - target_word_baseline) #[1, 768]

        igs[:,target_word_index,:] = target_word_igs #[1, 7, 768]

    attributions = igs.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    predicted_token = tokenizer.decode(predicted_token_id)
    print(f"Predicted Token: {predicted_token}")
    print("")
    for key, value in sorted(zip(all_tokens, attributions), key=lambda item: item[1], reverse=True):
        print(f"{key}: {value}")
    print("")

    return predicted_token, predicted_token_id, input_ids, attention_mask, attributions

def igs_nano_unvectorized(context: str, n_steps: int = 50) -> tuple[str, int, torch.Tensor]:
    print("===========================================")

    # tokenizer is the same as the one used for hf
    encoded_input = tokenizer(context, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())

    model_nano.eval()
    # with torch.no_grad():.....(not sure if I need this)
    # difference
    output_logits = model_nano(input_ids)[0].to(device)

    # difference
    next_token_logits = output_logits[0, 0, :].to(device)
    predicted_token_id = torch.argmax(next_token_logits).item()

    baseline_input_ids = torch.zeros_like(input_ids).to(device)
    baseline_embeddings = model_nano.transformer.wte(baseline_input_ids).to(device)

    igs = torch.zeros_like(baseline_embeddings)
    for target_word_index in range(igs.size(1)):
        target_word_baseline = baseline_embeddings[0, target_word_index, :]

        input_embeddings = model_nano.transformer.wte(input_ids)
        target_word_embedding = input_embeddings[0, target_word_index, :]

        position_ids = torch.arange(0, input_embeddings.size(1)).unsqueeze(0)
        position_embeddings = model_nano.transformer.wpe(position_ids)

        ig = torch.zeros_like(target_word_embedding)
        for step in range(n_steps):
            alpha = step / n_steps

            modified_embedding = target_word_baseline + alpha * (
                target_word_embedding - target_word_baseline
            )
            modified_embedding.requires_grad_(True)
            modified_embedding.retain_grad()

            if modified_embedding.grad is not None:
                modified_embedding.grad.zero_()

            step_embeddings = input_embeddings.clone()
            step_embeddings.requires_grad_(True)
            step_embeddings[0, target_word_index, :] = modified_embedding
            embeddings = step_embeddings + position_embeddings
            embeddings = model_nano.transformer.drop(embeddings)

            # difference
            for block in model_nano.transformer.h:
                embeddings = block(embeddings)

            embeddings = model_nano.transformer.ln_f(embeddings)
            output_at_step = model_nano.lm_head(embeddings)

            class_output_at_step = output_at_step[0, -1, predicted_token_id]
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as prof:

            class_output_at_step.backward(retain_graph=True)

            assert modified_embedding.grad is not None

            ig_at_step = modified_embedding.grad / n_steps
            ig += ig_at_step

        ig *= target_word_embedding - target_word_baseline
        igs[0, target_word_index, :] = ig

    predicted_token = tokenizer.decode(predicted_token_id)

    attributions = igs.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    print(f"Predicted Token: {predicted_token}")
    print("")
    for key, value in sorted(zip(all_tokens, attributions, strict=False), key=lambda item: item[1], reverse=True):
        print(f"{key}: {value}")
    print("")

    return predicted_token, predicted_token_id, input_ids, attention_mask, attributions
