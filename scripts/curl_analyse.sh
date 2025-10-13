#!/usr/bin/env bash
# Simple client to hit the /analyse endpoint
curl -X POST \
  -F "file=@sample_images/image_04.png" \
  http://127.0.0.1:8000/analyse | jq
