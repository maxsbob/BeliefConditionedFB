#!/usr/bin/env bash
# Create the virtual environment with uv.
# Proxy variables are removed for this command only, so packages are always downloaded
# directly from PyPI even if http(s)_proxy is exported for other tools (e.g. wandb).
set -euo pipefail
cd "$(dirname "$0")"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    uv sync "$@"
