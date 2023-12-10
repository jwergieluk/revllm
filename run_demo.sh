#!/bin/bash

set -e
set -u

cd "$( dirname "${BASH_SOURCE[0]}" )"
export PYTHONPATH='.'
streamlit run demo.py --theme.base=light --server.maxUploadSize 500 --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none
