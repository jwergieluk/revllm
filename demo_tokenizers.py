import pandas as pd
import regex
import streamlit as st
import tiktoken
from tiktoken import Encoding


def display_words_as_dataframe(words: list[str], num_columns: int = 7) -> None:
    """Displays a list of words as a dataframe with a given number of columns."""
    num_words = len(words)
    num_rows = (num_words + num_columns - 1) // num_columns
    words = [f'"{word}"' for word in words]
    words = words + [""] * (num_rows * num_columns - num_words)
    words = [words[i : i + num_columns] for i in range(0, len(words), num_columns)]
    words_df = pd.DataFrame(words)
    st.dataframe(words_df, hide_index=True, use_container_width=True)


def split_into_words_and_show(tokenizer: Encoding, text: str) -> None:
    regex_word_splitter = regex.compile(tokenizer._pat_str)
    words = regex_word_splitter.findall(text)
    st.write("The input text is split into the following words:")
    display_words_as_dataframe(words)


def show_tokens_word_by_word(tokenizer: Encoding, text: str) -> None:
    regex_word_splitter = regex.compile(tokenizer._pat_str)
    words = regex_word_splitter.findall(text)

    word_tokens_pairs = []
    for word in words:
        tokens = tokenizer.encode(word)
        word_tokens_pairs.append((word, tokens))

    max_num_tokens = max(len(tokens) for _, tokens in word_tokens_pairs)
    columns = ["Word", "Len", *[f"Token {i+1}" for i in range(max_num_tokens)]]
    dtype = ["string", "int", *["int" for i in range(max_num_tokens)]]
    word_tokens_df = pd.DataFrame(
        [
            (
                f'"{word}"',
                len(word),
                *tokens,
                *(
                    [
                        None,
                    ]
                    * (max_num_tokens - len(tokens))
                ),
            )
            for word, tokens in word_tokens_pairs
        ],
        columns=columns,
    )
    # for col, col_dtype in zip(columns[2:], dtype[2:]):
    #     word_tokens_df[col] = word_tokens_df[col].astype(col_dtype)
    st.write("The words are tokenized as follows:")
    st.dataframe(word_tokens_df, use_container_width=True)


def show_page_tokenizer(tokenizer_name: str):
    tokenizer = tiktoken.get_encoding(tokenizer_name)

    st.write(
        f'The tokenizer "{tokenizer_name}" splits the input text into tokens using '
        f"the following regular expression:"
    )
    st.code(tokenizer._pat_str, language="regex")

    if hasattr(tokenizer, "_special_tokens"):
        st.write("The tokenizer has the following special tokens:")
        special_tokens_df = pd.DataFrame(
            tokenizer._special_tokens.items(), columns=["Token", "Value"]
        )
        st.dataframe(special_tokens_df)

    st.write(f"The maximal token value is _{tokenizer.max_token_value}_.")

    default_input_str = (
        "Hello hello Hello, hello.Hello.  Hello. hellohello helloHello HelloHelloHello"
    )
    input_str = st.text_input("Input text for the tokenizer", default_input_str)
    input_str = str(input_str)
    if not input_str:
        return

    split_into_words_and_show(tokenizer, input_str)
    show_tokens_word_by_word(tokenizer, input_str)


def main():
    st.title("`tiktoken` Tokenizers Demo")
    all_encoding_names = tiktoken.list_encoding_names()
    selected_encoding = st.selectbox(
        "Encoding",
        all_encoding_names,
        index=0,
    )
    show_page_tokenizer(selected_encoding)


if __name__ == "__main__":
    main()
