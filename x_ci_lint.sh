#!/bin/bash
set -e
ruff format --check .
ruff check .
