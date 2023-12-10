#!/bin/bash

set -e

conda_add_channel() {
  channel_name="$1"
  if conda config --show channels | grep -q ${channel_name}
  then
      echo "Conda channel ${channel_name} is known already"
  else
      conda config --add channels ${channel_name}
  fi
}

conda_rm_env() {
  CONDA_ENV_NAME="${1}"
  if conda env list | grep -q ${CONDA_ENV_NAME}
  then
    echo "Remove existing conda env ${CONDA_ENV_NAME}"
    conda env remove -n "${CONDA_ENV_NAME}" -y
  fi
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source "${DIR}/init_conda.sh"
set -u  # init conda has some unbounded variables

CONDA_ENV_NAME="revllm"
CONDA_PACKAGE_FILES="--file conda_packages.txt"

conda_rm_env ${CONDA_ENV_NAME}
conda_add_channel conda-forge
conda config --set channel_priority strict

conda create -n "${CONDA_ENV_NAME}" --yes ${CONDA_PACKAGE_FILES}
conda activate "${CONDA_ENV_NAME}"

if [[ -f ".evn" ]]
then
    conda env config vars set "$(paste -d ' ' -s "${DIR}/.env" | sed -e 's/"//g')"
fi

pip install --upgrade -r pip_packages.txt
conda env list
