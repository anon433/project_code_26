#!/usr/bin/env bash
set -euo pipefail
reproduction_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Resolve caller-relative config paths before changing to the source directory.
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --config && $((i+1)) -lt ${#args[@]} ]]; then
    args[i+1]="$(realpath -- "${args[i+1]}")"
  elif [[ "${args[i]}" == --config=* ]]; then
    args[i]="--config=$(realpath -- "${args[i]#--config=}")"
  fi
done
cd -- "$reproduction_root"
export PYTHONPATH="$reproduction_root/src:$reproduction_root${PYTHONPATH:+:$PYTHONPATH}"
exec python -m src.reproduction.cli submit --kind masks "${args[@]}"
