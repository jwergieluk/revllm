from collections.abc import Sequence

import numpy as np
from wordcloud import WordCloud


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


def make_word_cloud(tokens: list[str], weights: Sequence[float], top_k: int = 100) -> np.ndarray:
    token_weight_dict = {t: weights[i] for i, t in enumerate(tokens)}
    wc = WordCloud(
        width=1200,
        height=800,
        normalize_plurals=False,
        include_numbers=True,
        colormap="Blues",
        max_words=top_k,
    ).generate_from_frequencies(token_weight_dict)
    return wc.to_image()
