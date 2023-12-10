import streamlit as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SUPPORTED_MODELS = ("", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

PAGE_MODEL_ARCHITECTURE = "Architecture"
PAGE_MODEL_CHAT = "Chat"
ALL_PAGES = (
    PAGE_MODEL_ARCHITECTURE,
    PAGE_MODEL_CHAT,
)


def main():
    st.markdown("# RevLLM")
    st.caption("Reverse Engineering Tools for Language Models")

    selected_model = st.sidebar.selectbox(
        "Select model",
        SUPPORTED_MODELS,
        index=0,
    )
    if not str(selected_model).strip():
        return

    with st.spinner("Loading model..."):
        config = AutoConfig.from_pretrained(selected_model)
        tokenizer = AutoTokenizer.from_pretrained(selected_model)
        model = AutoModelForCausalLM.from_pretrained(selected_model)
        model.eval()
    selected_page = st.sidebar.radio(
        "Select page",
        ALL_PAGES,
        index=0,
    )

    if selected_page == PAGE_MODEL_ARCHITECTURE:
        show_page_model_architecture(config, tokenizer, model)
    if selected_page == PAGE_MODEL_CHAT:
        show_page_generate(config, tokenizer, model)


def show_page_model_architecture(config, tokenizer, model):
    st.markdown("### Model Card")

    num_model_params = sum(p.nelement() for p in model.parameters())
    units = {
        "K": 1000,
        "M": 1000_000,
        "B": 1000_000_000,
    }
    num_model_params_in_unit = num_model_params
    for unit, divider in units.items():
        if num_model_params > divider:
            num_model_params_in_unit = f"{num_model_params / divider:.0f}{unit}"

    num_transformer_blocks = len(model.transformer.h)

    col0, col1 = st.columns(2)
    col0.metric("Model parameters", num_model_params_in_unit)
    col1.metric("Transformer Blocks", num_transformer_blocks)

    st.caption("Model Architecture")
    st.code(model)

    st.caption("Model config")
    st.code(config)

    st.caption("Model tokenizer")
    st.code(tokenizer)


def show_page_generate(config, tokenizer, model):
    st.markdown("## Generate")
    input_text = st.text_input("Input text", "Hi.")
    if not str(input_text).strip():
        return

    encoded_input = tokenizer(str(input_text), return_tensors="pt")
    output = model.generate(**encoded_input, max_length=60)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.caption("Generated text")
    st.code(generated_text)


if __name__ == "__main__":
    main()
