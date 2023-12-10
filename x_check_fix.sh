#!/bin/bash
set -e
export PYTHONPATH='.'
ruff format .
ruff check --fix .
pytest -v test
