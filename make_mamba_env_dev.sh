#!/bin/bash

set -e
set -u

MAMBA="${HOME}/.local/bin/micromamba"
MAMBA_ROOT_PREFIX="${HOME}/micromamba"

if [ ! -f "${MAMBA}" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    mkdir -p $(dirname ${MAMBA})
    mv bin/micromamba ${MAMBA}
    rm -rf bin
fi

${MAMBA} self-update --yes
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ENV_NAME="revllm_dev"
CONDA_PACKAGE_FILES="--file conda_packages.txt --file conda_packages_dev.txt"
$MAMBA create -n "${ENV_NAME}" --yes ${CONDA_PACKAGE_FILES}
