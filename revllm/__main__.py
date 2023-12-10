import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@click.group()
def cli():
    pass


@cli.command()
def hello():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Force model to use CPU
    model.to("cpu")

    input_text = "Hello, how are you?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(input_ids, max_length=50)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    click.echo(output_text)


if __name__ == "__main__":
    cli()
