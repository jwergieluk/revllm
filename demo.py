import pandas as pd
import streamlit as st


def main():
    st.markdown("# RevLLM")
    uploaded_file = st.sidebar.file_uploader("Upload a file for analysis")

    selected_page = st.sidebar.radio(
        "Select page",
        ('Welcome Page',),
        index=0,
    )


if __name__ == '__main__':
    main()
