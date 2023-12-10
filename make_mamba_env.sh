#!/bin/bash

set -e
set -u

MAMBA="${HOME}/.local/bin/micromamba"
MAMBA_ROOT_PREFIX="${HOME}/.mamba"

if [ ! -f "${MAMBA}" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj micromamba
fi


. "${MAMBA_ROOT_PREFIX}/etc/profile.d/micromamba.sh"
${MAMBA} self-update --yes


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ENV_NAME="woyb"
CONDA_PACKAGE_FILES="--file conda_packages.txt"

$MAMBA create -n "${ENV_NAME}" --yes ${CONDA_PACKAGE_FILES}
$MAMBA activate "${ENV_NAME}"
pip install --upgrade -r pip_packages.txt
$MAMBA env list
