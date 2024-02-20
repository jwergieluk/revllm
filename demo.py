import os
from operator import itemgetter

import altair as alt
import numpy as np
import pandas as pd
import psutil
import scipy.stats
import streamlit as st
import torch
from streamlit_extras.word_importances import format_word_importances

from demo_tokenizers import display_words_as_dataframe, show_page_tokenizer
from revllm.helpers import make_word_cloud, reformat_lines
from revllm.model_wrapper import (
    ModelWrapper,
    get_final_predictions_ranks,
    get_top_k_intersection_scores,
)
from revllm.prompts import get_daily_prompts

APP_TITLE = "RevLLM: Reverse Engineering Tools for Language Models"
SUPPORTED_MODELS = ("", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
AVAILABLE_DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

PAGE_DOCS = "Documentation"
PAGE_MODEL_ARCHITECTURE = "Architecture"
PAGE_TOKENIZER = "Tokenizer"
PAGE_TOKEN_EMBEDDINGS = "Token Embeddings"
PAGE_PROMPT_IMPORTANCE = "Prompt Importance"
PAGE_LOGIT_LENS = "Logit Lens"
PAGE_CIRCUIT_DISCOVERY = "Circuit Discovery"
PAGE_GENERATE = "Generate"
PAGE_ATTENTIONS = "Multi-Head Self-Attention"
ALL_PAGES = (
    PAGE_DOCS,
    PAGE_MODEL_ARCHITECTURE,
    PAGE_TOKENIZER,
    PAGE_TOKEN_EMBEDDINGS,
    PAGE_GENERATE,
    PAGE_PROMPT_IMPORTANCE,
    PAGE_LOGIT_LENS,
    PAGE_ATTENTIONS,
    # PAGE_CIRCUIT_DISCOVERY,
)

IMPORTANCE_INTEGRATED_GRADIENTS = "Integrated Gradients"
IMPORTANCE_SEQ_INTEGRATED_GRADIENTS = "Sequential Integrated Gradients"
IMPORTANCE_LIME = "LIME"
ALL_IMPORTANCE_METHODS = (
    IMPORTANCE_INTEGRATED_GRADIENTS,
    IMPORTANCE_SEQ_INTEGRATED_GRADIENTS,
)

st.set_page_config(page_title=APP_TITLE, page_icon=":microscope:")


def display_markdown_file(md_file_name: str) -> None:
    with open(md_file_name) as f:
        md_file = f.read()

    # skip YAML preamble
    if md_file.startswith("---"):
        lines = []
        yaml_block_delimiter_count = 0
        for line in md_file.split("\n"):
            if line.startswith("---"):
                yaml_block_delimiter_count += 1
                if yaml_block_delimiter_count <= 2:
                    continue
            if yaml_block_delimiter_count < 2:
                continue
            lines.append(line)
        md_file = "\n".join(lines)

    st.markdown(md_file, unsafe_allow_html=True)


def display_vega_bar_chart(data: pd.DataFrame, description: str = "") -> None:
    if data.empty:
        return
    category_col: str = data.columns[0]
    value_col: str = data.columns[1]
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": description,
        "encoding": {
            "y": {"field": category_col, "type": "nominal"},
            "x": {"field": value_col, "type": "quantitative"},
        },
        "layer": [
            {"mark": "bar"},
            {
                "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 3},
                "encoding": {"text": {"field": value_col, "type": "quantitative"}},
            },
        ],
    }
    st.vega_lite_chart(data, spec, use_container_width=True)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)


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
        display_markdown_file("docs/gpt-2.md")
        return

    model_wrapper = get_model_wrapper(selected_model, device_name=device)
    selected_page = st.sidebar.radio(
        "Select page",
        ALL_PAGES,
        index=0,
    )
    st.sidebar.caption(f"Memory usage: {get_memory_usage():.0f} MB")

    if selected_page == PAGE_DOCS:
        display_markdown_file("docs/gpt-2.md")
    if selected_page == PAGE_MODEL_ARCHITECTURE:
        show_page_model_architecture(model_wrapper)
    if selected_page == PAGE_TOKENIZER:
        show_page_tokenizer("gpt2")
    if selected_page == PAGE_TOKEN_EMBEDDINGS:
        show_page_token_embeddings(model_wrapper)
    if selected_page == PAGE_PROMPT_IMPORTANCE:
        show_page_prompt_importance(model_wrapper)
    if selected_page == PAGE_LOGIT_LENS:
        show_page_logit_lens(model_wrapper)
    if selected_page == PAGE_GENERATE:
        show_page_generate(model_wrapper)
    if selected_page == PAGE_ATTENTIONS:
        show_page_attentions(model_wrapper)


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


def get_prompt(default_prompt: str = "") -> str:
    sample_prompts = [""] + get_daily_prompts(20)
    selected_sample_prompt = st.selectbox("Sample prompts", sample_prompts)
    prompt = st.text_input("User prompt", default_prompt)

    prompt = str(prompt).strip()
    prompt = prompt if prompt else selected_sample_prompt
    return prompt


def show_page_generate(wrapper: ModelWrapper):
    st.header("Generate")

    prompt = get_prompt()
    temperature = st.slider("Temperature", min_value=0.1, max_value=10.0, value=0.9, step=0.1)
    st.write("Lower temperatures make the model more deterministic.  Higher temperatures increase "
             "variability.")
    checkbox_reformat_output = st.checkbox("Reformat output", value=True)
    max_new_tokens = st.number_input("Number of new tokens", min_value=1, max_value=1000, value=35)

    if not prompt:
        return

    button_run = st.button("Generate")
    if not button_run:
        return

    with st.spinner("Evaluating model..."):
        generated_text = wrapper.generate(prompt, max_new_tokens, temperature)

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=checkbox_skip_special_tokens)
    if checkbox_reformat_output:
        generated_text = reformat_lines(generated_text, max_line_len=80)
    st.caption("Generated text")
    st.code(generated_text, language="text")


def show_page_logit_lens(wrapper: ModelWrapper, k: int = 50):
    st.header("Logit Lens")
    st.write("Employing "
             "[logits lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), "
             "we are able to visualize the model's state at every hidden layer h.")
    st.header("Choose a prompt")
    prompt = get_prompt("The capital of Japan is the city of ")

    if not prompt:
        return

    button_run = st.button("Run Logit Lens")
    if not button_run:
        return

    logit_lens_data = wrapper.run_logit_lens(prompt)

    st.write(
        f'Output token: _"{logit_lens_data.output_token}"_ ({logit_lens_data.output_token_ids})'
    )
    df = logit_lens_data.hidden_state_most_likely_token_df
    max_logits_df = pd.DataFrame(logit_lens_data.hidden_state_max_logits, columns=df.columns)
    descriptive_index = [f"h_{i}_out" if len(str(i)) > 1 else f"h_0{i}_out" for i in df.index]
    df.index = descriptive_index
    max_logits_df.index = descriptive_index
    st.caption("Tokens")

    st.dataframe(df.style.background_gradient(cmap="Blues", axis=None, gmap=max_logits_df))

    if len(df.columns) > 1:
        random_token = np.random.choice(df.columns[1:])
    else:
        random_token = df.columns[0]
    random_token_index = list(df.columns).index(random_token)
    random_sub_prompt = df.columns[: random_token_index + 1].tolist()
    random_sub_prompt = [token[token.index("_") + 1 :] for token in random_sub_prompt]
    random_sub_prompt = "".join(random_sub_prompt)
    random_df_index = np.random.choice(df.index)
    st.write(
        "At each token, the model considers the sub-prompt up to it. "
        "It generates next-token prediction for this sub-prompt. The final output token is "
        "simply the last of these predictions. All dataframes on this page are populated "
        "according to this fact.  \n\nFor example:"
    )
    st.write(f"Random token: {random_token}")
    st.write(f"Sub-prompt: {random_sub_prompt}")
    st.write(f"Predicted token at layer {random_df_index}: {df.loc[random_df_index, random_token]}")

    st.caption("Logits")
    st.dataframe(max_logits_df.style.background_gradient(cmap="Blues", axis=None))
    st.write(
        "Logits at each layer refer to the raw output of the model before conversion to probability "
        "scores. Here we include the maximum logit value at each layer."
    )
    st.caption(f"Top {k} intersection scores")
    top_k_intersection_scores = get_top_k_intersection_scores(
        probabilities_tensor=logit_lens_data.hidden_state_probabilities, k=k
    )
    top_k_intersection_scores_df = pd.DataFrame(
        data=top_k_intersection_scores.squeeze(-1), columns=df.columns, index=df.index
    )
    st.dataframe(top_k_intersection_scores_df.style.background_gradient(cmap="Blues", axis=None))

    st.write(
        "The [top k intersection score](https://arxiv.org/pdf/2305.13417.pdf) of two probability "
        "tensors measures the degree to which their top k predicted indices overlap (0 = no overlap, "
        "1 = full overlap). Here, at every layer we compare its ouptut probability tensor with "
        "the final layer's output probability tensor."
        ""
    )

    st.caption(f"Top {k} intersection scores as a line chart")
    st.line_chart(top_k_intersection_scores_df)

    st.caption("Final prediction ranks")
    final_prediction_ranks = get_final_predictions_ranks(
        probabilities_tensor=logit_lens_data.hidden_state_probabilities
    )
    final_prediction_ranks_df = pd.DataFrame(
        data=final_prediction_ranks.squeeze(-1), columns=df.columns, index=df.index
    )
    st.dataframe(final_prediction_ranks_df.style.background_gradient(cmap="Blues", axis=None))

    st.write(
        "The final prediction for each sub-prompt refers to the predicted (top ranked) token at the "
        "final layer (the bottom row of the first dataframe above). Here, at each layer, we view the "
        "final prediction's rank among all tokens."
    )

    st.caption("Final prediction ranks as a line chart")
    st.line_chart(final_prediction_ranks_df)

    st.subheader("Word clouds from token logits")
    st.write(
        "At each layer, we include a word cloud for the full prompt's top 20 predicted "
        "tokens, with a probabilities histogram included."
    )

    all_tokens = wrapper.tokenizer.get_all_tokens()
    for i, layer in enumerate(descriptive_index):
        st.write(f"Layer **{layer}**")

        layer_probabilities: np.ndarray = (
            logit_lens_data.hidden_state_probabilities[i, -1].squeeze().numpy()
        )
        p_cutoff = np.sort(layer_probabilities)[-20]

        top_tokens = [
            (all_tokens[i], p) for i, p in enumerate(layer_probabilities) if p >= p_cutoff
        ]
        top_tokens = sorted(top_tokens, key=itemgetter(1), reverse=True)
        top_tokens_df = pd.DataFrame(data=top_tokens, columns=["token", "probability"])

        word_cloud_array = make_word_cloud(all_tokens, layer_probabilities)

        st.image(word_cloud_array)
        display_vega_bar_chart(top_tokens_df, description=f"Top tokens in layer {layer}")


def show_page_prompt_importance(wrapper: ModelWrapper):
    st.header("Prompt Importance Analysis")

    selected_importance_method = st.selectbox(
        "Select token importance scoring method",
        ALL_IMPORTANCE_METHODS,
        index=0,
    )

    sample_prompts = [""] + get_daily_prompts(20)
    selected_sample_prompt = st.selectbox("Sample prompts", sample_prompts)
    prompt = st.text_input("User prompt", "")

    prompt = str(prompt).strip()
    prompt = prompt if prompt else selected_sample_prompt
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


def show_page_attentions(wrapper: ModelWrapper):
    st.header(PAGE_ATTENTIONS)
    prompt = get_prompt("The capital of Japan is the city of ")

    if not prompt:
        return

    button_generate = st.button("Generate")
    if not button_generate:
        return

    input_ids = wrapper.tokenizer.encode(prompt)
    input_tokens = input_ids[0].tolist()
    tokens_list = [wrapper.tokenizer.decode([input_token]) for input_token in input_tokens]
    outputs = wrapper.model.forward_with_diagnostics(input_ids)
    attentions = outputs.attentions

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    indexed_tokens_list = [
        "0" + str(index) + "_" + token if len(str(index)) == 1 else str(index) + "_" + token
        for index, token in enumerate(tokens_list)
    ]
    st.write(f"Input tokens: {indexed_tokens_list}")
    for layer in list(range(num_layers)):
        st.markdown(
            f"<h3 style='text-align: center; text-decoration: underline;'><b>Layer {layer + 1}</b></h3>",
            unsafe_allow_html=True,
        )
        charts = []
        for head in list(range(num_heads)):
            attention_df = pd.DataFrame(
                data=attentions[layer][0, head, :, :],
                columns=indexed_tokens_list,
                index=indexed_tokens_list,
            )

            attention_df_long = attention_df.reset_index().melt(
                id_vars=["index"], var_name="target", value_name="attn"
            )

            color_scale = alt.Scale(
                domain=[0, attention_df_long["attn"].max()],
                range=["white", "#dceefb", "#99ccf0", "#5599e2", "#2a7bce", "#005bb5"],
            )

            # Now use attention_df_long for the Altair chart
            attention_chart = (
                alt.Chart(attention_df_long)
                .mark_rect()
                .encode(
                    x=alt.X("target:O", title="", axis=None),
                    y=alt.Y("index:O", title="", axis=None),
                    color=alt.Color(
                        "attn:Q",
                        scale=color_scale,  # lt.Scale(domain=[0, attention_df_long['attn'].max()], range=['white', "blue"])
                        legend=None,
                    ),
                )
            )
            attention_chart_with_title = attention_chart.properties(title=f"Head {head + 1}")
            charts.append(attention_chart_with_title)

            # st.altair_chart(attention_chart, use_container_width=True)
        rows = [
            alt.hconcat(*charts[i : i + 4]) for i in range(0, len(charts), 4)
        ]  # Create rows of up to 4 charts
        grid = alt.vconcat(*rows)  # Concatenate rows vertically

        st.altair_chart(grid, use_container_width=True)
        st.write("")


if __name__ == "__main__":
    main()
