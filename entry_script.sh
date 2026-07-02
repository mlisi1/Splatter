#!/bin/bash
set -euo pipefail
exec python3 /workspace/pipeline.py "$@"
