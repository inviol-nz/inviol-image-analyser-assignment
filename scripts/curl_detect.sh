#!/usr/bin/env bash
set -euo pipefail
IMG=${1:-sample_images/image_04.png}
curl -s -X POST \
  -F "file=@${IMG}" \
  http://127.0.0.1:8000/debug/detect | jq
