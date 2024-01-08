import os

import psutil
import streamlit as st
import torch

from revllm.model_wrapper import ModelWrapper

APP_TITLE = "RevLLM: Reverse Engineering Tools for Language Models"
SUPPORTED_MODELS = ("", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
AVAILABLE_DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

PAGE_MODEL_ARCHITECTURE = "Architecture"
PAGE_GENERATE = "Generate"
ALL_PAGES = (
    PAGE_MODEL_ARCHITECTURE,
    PAGE_GENERATE,
)

st.set_page_config(page_title=APP_TITLE, page_icon=":rocket:")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)


def reformat_lines(input_string: str, max_line_len: int):
    # Split the input string into lines
    lines = input_string.splitlines()

    # Initialize an empty list to store the reformatted lines
    reformatted_lines = []

    for line in lines:
        # Split the line into words
        words = line.split()

        # Initialize variables to store the current line and its length
        current_line = []
        current_line_len = 0

        for word in words:
            # Check if adding the current word to the current line exceeds the max_line_len
            if current_line_len + len(word) + len(current_line) > max_line_len:
                # Add the current line to the reformatted lines list
                reformatted_lines.append(" ".join(current_line))
                # Reset the current line and its length
                current_line = []
                current_line_len = 0

            # Add the current word to the current line
            current_line.append(word)
            current_line_len += len(word)

        # Add the remaining part of the current line to the reformatted lines list
        reformatted_lines.append(" ".join(current_line))

    # Join the reformatted lines to form the final reformatted string
    reformatted_string = "\n".join(reformatted_lines)

    return reformatted_string


@st.cache_resource(show_spinner="Loading model...")
def get_model_wrapper(
    model_name: str,
    device_name: str = "cpu",
    compiled: bool = False,
) -> ModelWrapper:
    return ModelWrapper(model_name=model_name, device_name=device_name, compiled=compiled)


def main():
    st.markdown("# RevLLM")
    st.caption("Reverse Engineering Tools for Language Models")

    selected_device = st.sidebar.selectbox(
        "Select device",
        AVAILABLE_DEVICES,
        index=0,
    )
    device = str(selected_device).strip()
    selected_model = st.sidebar.selectbox(
        "Select model",
        SUPPORTED_MODELS,
        index=0,
    )
    if not str(selected_model).strip():
        return

    model_wrapper = get_model_wrapper(selected_model, device_name=device)
    selected_page = st.sidebar.radio(
        "Select page",
        ALL_PAGES,
        index=0,
    )
    st.sidebar.caption(f"Memory usage: {get_memory_usage():.0f} MB")

    if selected_page == PAGE_MODEL_ARCHITECTURE:
        show_page_model_architecture(model_wrapper)
    if selected_page == PAGE_GENERATE:
        show_page_generate(model_wrapper)


def show_page_model_architecture(wrapper: ModelWrapper):
    st.markdown("### Model Card")

    num_model_params = sum(p.nelement() for p in wrapper.model.parameters())
    units = {
        "K": 1000,
        "M": 1000_000,
        "B": 1000_000_000,
    }
    num_model_params_in_unit = num_model_params
    for unit, divider in units.items():
        if num_model_params > divider:
            num_model_params_in_unit = f"{num_model_params / divider:.0f}{unit}"

    num_transformer_blocks = len(wrapper.model.transformer.h)

    col0, col1 = st.columns(2)
    col0.metric("Model parameters", num_model_params_in_unit)
    col1.metric("Transformer Blocks", num_transformer_blocks)

    st.caption("Model Architecture")
    st.code(str(wrapper))

    # st.caption("Model config")
    # st.code(config)
    #
    # st.caption("Model tokenizer")
    # st.code(tokenizer)


def show_page_generate(wrapper: ModelWrapper):
    st.markdown("## Generate")
    input_text = st.text_input("Input text", "Hello, my name is")
    checkbox_skip_special_tokens = st.checkbox("Skip special tokens", value=True)
    checkbox_reformat_output = st.checkbox("Reformat output", value=True)
    input_output_len = st.number_input("Output length", min_value=1, max_value=1000, value=100)
    if not str(input_text).strip():
        return

    button_generate = st.button("Generate")
    if not button_generate:
        return

    with st.spinner("Evaluating model..."):
        generated_text = wrapper.generate(input_text)

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=checkbox_skip_special_tokens)
    if checkbox_reformat_output:
        generated_text = reformat_lines(generated_text, max_line_len=80)
    st.caption("Generated text")
    st.code(generated_text, language="text")


if __name__ == "__main__":
    main()
