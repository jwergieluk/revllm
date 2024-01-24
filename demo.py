import os

import numpy as np
import pandas as pd
import psutil
import scipy.stats
import streamlit as st
import torch
from streamlit_extras.word_importances import format_word_importances

from demo_tokenizers import display_words_as_dataframe, show_page_tokenizer
from revllm.model_wrapper import ModelWrapper

APP_TITLE = "RevLLM: Reverse Engineering Tools for Language Models"
SUPPORTED_MODELS = ("", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
AVAILABLE_DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

PAGE_MODEL_ARCHITECTURE = "Architecture"
PAGE_TOKENIZER = "Tokenizer"
PAGE_TOKEN_EMBEDDINGS = "Token Embeddings"
PAGE_PROMPT_IMPORTANCE = "Prompt Importance"
PAGE_CIRCUIT_DISCOVERY = "Circuit Discovery"
PAGE_GENERATE = "Generate"
ALL_PAGES = (
    PAGE_MODEL_ARCHITECTURE,
    PAGE_TOKENIZER,
    PAGE_TOKEN_EMBEDDINGS,
    PAGE_GENERATE,
    PAGE_PROMPT_IMPORTANCE,
    PAGE_CIRCUIT_DISCOVERY,
)

IMPORTANCE_INTEGRATED_GRADIENTS = "Integrated Gradients"
IMPORTANCE_LIME = "LIME"
ALL_IMPORTANCE_METHODS = (
    IMPORTANCE_INTEGRATED_GRADIENTS,
    IMPORTANCE_LIME,
)

st.set_page_config(page_title=APP_TITLE, page_icon=":microscope:")


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
    return ModelWrapper(model_name=model_name, device_type=device_name, compiled=compiled)


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
    if selected_page == PAGE_TOKENIZER:
        show_page_tokenizer("gpt2")
    if selected_page == PAGE_TOKEN_EMBEDDINGS:
        show_page_token_embeddings(model_wrapper)
    if selected_page == PAGE_PROMPT_IMPORTANCE:
        show_page_prompt_importance(model_wrapper)
    if selected_page == PAGE_GENERATE:
        show_page_generate(model_wrapper)


def show_page_model_architecture(wrapper: ModelWrapper):
    st.header("Model Card")

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

    col0, col1, col2, col3 = st.columns(4)
    col0.metric("Model parameters", num_model_params_in_unit)
    col1.metric("Transformer Blocks", num_transformer_blocks)
    col2.metric("Vocabulary size", wrapper.model.get_vocab_size())
    col3.metric("Block size", wrapper.model.get_block_size())

    st.caption("Model Architecture")
    st.code(str(wrapper))


@st.cache_resource(show_spinner="Calculating embedding dimension statistics...")
def get_dim_stats_df(model_name: str, weights: np.ndarray) -> pd.DataFrame:
    dim_mean = weights.mean(axis=0)
    dim_std = weights.std(axis=0)
    dim_kurt = scipy.stats.kurtosis(weights, axis=0)
    dim_stats_df = pd.DataFrame(
        [(dim_mean[i], dim_std[i], dim_kurt[i]) for i in range(weights.shape[1])],
        columns=["Mean", "Std", "Kurtosis"],
    )
    return dim_stats_df


@st.cache_resource(show_spinner="Calculating embedding matrix statistics...")
def get_weight_stats_df(model_name: str, weights: np.ndarray) -> pd.DataFrame:
    weights_999_quantile = np.quantile(weights, 0.999)
    weights_001_quantile = np.quantile(weights, 0.001)
    weight_stats = [
        ("Vocab size", weights.shape[0]),
        ("Embedding dimension", weights.shape[1]),
        ("Min", weights.min()),
        ("Max", weights.max()),
        ("Mean", weights.mean()),
        ("Std", weights.std()),
        ("0.001 quantile", weights_001_quantile),
        ("0.999 quantile", weights_999_quantile),
    ]
    weight_stats_df = pd.DataFrame(weight_stats, columns=["Metric", "Value"])
    return weight_stats_df


@st.cache_resource(show_spinner="Calculating standardized embedding matrix...")
def get_standardized_weights(model_name: str, weights: np.ndarray) -> np.ndarray:
    weights_range = np.quantile(weights, 0.999) - np.quantile(weights, 0.001)
    weights_standardized_01 = np.clip(0.5 + weights / (2.0 * weights_range), 0.0, 1.0)
    return weights_standardized_01


def show_page_token_embeddings(wrapper: ModelWrapper):
    st.header("Token Embeddings")
    weights = wrapper.model.transformer.wte.weight.data.cpu().numpy()

    st.subheader("Embedding matrix statistics")
    weight_stats_df = get_weight_stats_df(wrapper.model_name, weights)
    st.dataframe(weight_stats_df, use_container_width=False)

    st.subheader("Embedding dimension statistics")
    checkbox_show_dim_stats = st.checkbox("Show embedding dimension statistics", value=False)
    if checkbox_show_dim_stats:
        dim_stats_df = get_dim_stats_df(wrapper.model_name, weights)
        st.caption("Embedding dimension mean.")
        st.line_chart(dim_stats_df["Mean"], use_container_width=True, color="#246e69")
        st.caption("Embedding dimension standard deviation.")
        st.line_chart(dim_stats_df["Std"], use_container_width=True, color="#15799e")
        st.caption("Embedding dimension kurtosis.")
        st.line_chart(dim_stats_df["Kurtosis"], use_container_width=True, color="#e6a400")

    st.subheader("Embedding matrix entries")
    weights_standardized_01 = get_standardized_weights(wrapper.model_name, weights)
    vocab_size = weights.shape[0]
    embedding_dimension = weights.shape[1]
    st.write(
        f"The embedding matrix has {vocab_size} rows (one for each token) and "
        f"{embedding_dimension} columns (one for each embedding dimension). "
        "For the following plot, we standardized the embedding matrix to the range [0, 1] "
        "using the 0.01-0.99 inter-quantile range."
    )
    col0, col1 = st.columns(2)
    first_row = col0.number_input(
        "First row", min_value=0, max_value=vocab_size - 1, value=0, step=1, key="first_row"
    )
    first_row = int(first_row)
    num_rows = col1.number_input(
        "Number of rows",
        min_value=20,
        max_value=vocab_size - 1,
        value=1000,
        step=1,
        key="last_row",
    )
    button_display_embedding_weight = st.button("Display embedding weights")
    if button_display_embedding_weight:
        last_row = min(first_row + num_rows, vocab_size - 1)
        if abs(last_row - first_row) < 20:
            st.error("The number of rows to display must be at most 20.")
            return
        with st.spinner("Plotting embedding matrix..."):
            st.image(
                weights_standardized_01[first_row:last_row, :],
                caption="Standardized embedding weights",
                use_column_width=True,
            )

        if num_rows < 1001:
            tokens = wrapper.tokenizer.decode_tokens_separately(list(range(first_row, last_row)))
            st.caption("Tokens")
            display_words_as_dataframe(tokens, num_columns=10, hide_index=False)
        else:
            st.caption("Too many tokens to display.")


def show_page_generate(wrapper: ModelWrapper):
    st.markdown("## Generate")
    prompt = st.text_input("Input text", "Hello, my name is")
    checkbox_reformat_output = st.checkbox("Reformat output", value=True)
    max_new_tokens = st.number_input("Number of new tokens", min_value=1, max_value=1000, value=250)
    temperature = st.slider("Temperature", min_value=0.1, max_value=10.0, value=0.9, step=0.1)
    if not str(prompt).strip():
        return

    button_generate = st.button("Generate")
    if not button_generate:
        return

    with st.spinner("Evaluating model..."):
        generated_text = wrapper.generate(prompt, max_new_tokens, temperature)

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=checkbox_skip_special_tokens)
    if checkbox_reformat_output:
        generated_text = reformat_lines(generated_text, max_line_len=80)
    st.caption("Generated text")
    st.code(generated_text, language="text")


def show_page_prompt_importance(wrapper: ModelWrapper):
    st.header("Prompt Importance Analysis")

    selected_importance_method = st.selectbox(
        "Select token importance scoring method",
        ALL_IMPORTANCE_METHODS,
        index=0,
    )
    prompt = st.text_input("Prompt", "Hello, my name is")
    prompt = str(prompt).strip()
    max_new_tokens = st.number_input(
        "Number of new tokens to generate", min_value=1, max_value=100, value=10
    )
    checkbox_show_scores = st.checkbox("Show details", value=True)
    if not str(prompt).strip():
        return
    button_generate = st.button("Generate")
    if not button_generate:
        return

    scores_generator = None
    if selected_importance_method == IMPORTANCE_INTEGRATED_GRADIENTS:
        scores_generator = wrapper.yield_importance_integrated_gradients(prompt)
    if selected_importance_method == IMPORTANCE_LIME:
        scores_generator = wrapper.yield_importance_lime(prompt)
    if not scores_generator:
        return

    scores = []
    for i in range(max_new_tokens):
        score = next(scores_generator)
        scores.append(score)
        st.subheader(
            f"Generated Token {i + 1}: '{score.output_token}' (id: {score.output_token_id})"
        )

        st.caption("Input tokens with scores")
        html = format_word_importances(
            words=score.input_tokens,
            importances=score.input_token_scores,
        )
        st.write(html, unsafe_allow_html=True)

        if checkbox_show_scores:
            st.caption("Full importance score data")
            st.dataframe(score.get_input_score_df())


if __name__ == "__main__":
    main()
